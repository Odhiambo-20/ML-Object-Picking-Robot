#include "motion_planning/inverse_kinematics.hpp"
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/frames_io.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <urdf/model.h>
#include <Eigen/Dense>
#include <memory>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>

namespace motion_planning {

InverseKinematics::InverseKinematics(const std::string& robot_description)
    : nh_("inverse_kinematics")
{
    RCLCPP_INFO(nh_.get_logger(), "Initializing KDL Inverse Kinematics Solver");
    
    // Load URDF model
    urdf::Model robot_model;
    if (!robot_model.initString(robot_description)) {
        RCLCPP_FATAL(nh_.get_logger(), "Failed to parse URDF model");
        throw std::runtime_error("Failed to parse URDF model");
    }
    
    // Extract KDL tree from URDF
    KDL::Tree kdl_tree;
    if (!kdl_parser::treeFromUrdfModel(robot_model, kdl_tree)) {
        RCLCPP_FATAL(nh_.get_logger(), "Failed to extract KDL tree from URDF");
        throw std::runtime_error("Failed to extract KDL tree from URDF");
    }
    
    // Define chain (from base to end effector)
    std::string root_link = "base_link";
    std::string tip_link = "end_effector_link";
    
    if (!kdl_tree.getChain(root_link, tip_link, robot_chain_)) {
        RCLCPP_FATAL(nh_.get_logger(), "Failed to get chain from %s to %s", 
                     root_link.c_str(), tip_link.c_str());
        throw std::runtime_error("Failed to get KDL chain");
    }
    
    RCLCPP_INFO(nh_.get_logger(), "Robot chain has %d segments and %d joints", 
                robot_chain_.getNrOfSegments(), robot_chain_.getNrOfJoints());
    
    // Initialize solvers
    initializeSolvers();
    
    // Initialize joint limits
    initializeJointLimits(robot_model);
    
    // Initialize performance monitoring
    total_calls_ = 0;
    total_time_ = 0.0;
    
    RCLCPP_INFO(nh_.get_logger(), "KDL Inverse Kinematics initialized successfully");
}

void InverseKinematics::initializeSolvers() {
    // Forward kinematics solver
    fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(robot_chain_);
    
    // Inverse kinematics velocity solver (for Jacobian-based methods)
    ik_vel_solver_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(
        robot_chain_, 
        0.0001,  // eps
        150      // maxiter
    );
    
    // Levenberg-Marquardt position solver (most robust for industrial applications)
    ik_pos_solver_lma_ = std::make_unique<KDL::ChainIkSolverPos_LMA>(
        robot_chain_,
        1e-5,    // eps
        1000,    // maxiter
        1e-10    // eps_joints
    );
    
    // Newton-Raphson position solver (faster but less robust)
    ik_pos_solver_nr_ = std::make_unique<KDL::ChainIkSolverPos_NR>(
        robot_chain_,
        *fk_solver_,
        *ik_vel_solver_,
        1000,    // maxiter
        1e-6     // eps
    );
}

void InverseKinematics::initializeJointLimits(const urdf::Model& robot_model) {
    joint_names_.clear();
    joint_lower_limits_.clear();
    joint_upper_limits_.clear();
    joint_velocity_limits_.clear();
    joint_effort_limits_.clear();
    
    // Extract joint limits from URDF
    for (size_t i = 0; i < robot_chain_.getNrOfJoints(); ++i) {
        const KDL::Joint& joint = robot_chain_.getSegment(i).getJoint();
        
        if (joint.getType() != KDL::Joint::JointType::None) {
            std::string joint_name = joint.getName();
            joint_names_.push_back(joint_name);
            
            // Get limits from URDF
            urdf::JointConstSharedPtr urdf_joint = robot_model.getJoint(joint_name);
            if (urdf_joint && urdf_joint->limits) {
                joint_lower_limits_.push_back(urdf_joint->limits->lower);
                joint_upper_limits_.push_back(urdf_joint->limits->upper);
                joint_velocity_limits_.push_back(urdf_joint->limits->velocity);
                joint_effort_limits_.push_back(urdf_joint->limits->effort);
            } else {
                // Default limits if not specified
                joint_lower_limits_.push_back(-M_PI);
                joint_upper_limits_.push_back(M_PI);
                joint_velocity_limits_.push_back(2.0);  // rad/s
                joint_effort_limits_.push_back(100.0);  // Nm
            }
        }
    }
    
    num_joints_ = joint_names_.size();
    
    RCLCPP_INFO(nh_.get_logger(), "Initialized %zu joints with limits", num_joints_);
}

bool InverseKinematics::solveIK(const geometry_msgs::msg::Pose& target_pose,
                                const std::vector<double>& seed_state,
                                std::vector<double>& solution,
                                IKMethod method) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Convert ROS pose to KDL frame
    KDL::Frame target_frame;
    tf2::fromMsg(target_pose, target_frame);
    
    // Convert seed state to KDL JntArray
    KDL::JntArray seed_array(num_joints_);
    for (size_t i = 0; i < num_joints_; ++i) {
        seed_array(i) = (i < seed_state.size()) ? seed_state[i] : 0.0;
    }
    
    // Prepare solution array
    KDL::JntArray solution_array(num_joints_);
    
    int ik_result = -1;
    std::string method_name;
    
    // Solve using selected method
    switch (method) {
        case IKMethod::LMA:
            method_name = "Levenberg-Marquardt";
            ik_result = ik_pos_solver_lma_->CartToJnt(seed_array, target_frame, solution_array);
            break;
            
        case IKMethod::NR:
            method_name = "Newton-Raphson";
            ik_result = ik_pos_solver_nr_->CartToJnt(seed_array, target_frame, solution_array);
            break;
            
        case IKMethod::ITERATIVE:
            method_name = "Iterative";
            ik_result = solveIKIterative(target_frame, seed_array, solution_array);
            break;
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    
    // Update performance statistics
    total_calls_++;
    total_time_ += elapsed_ms;
    
    if (ik_result >= 0) {
        // Copy solution
        solution.resize(num_joints_);
        for (size_t i = 0; i < num_joints_; ++i) {
            solution[i] = solution_array(i);
        }
        
        // Check joint limits
        if (!checkJointLimits(solution)) {
            RCLCPP_WARN(nh_.get_logger(), "IK solution violates joint limits");
            return false;
        }
        
        // Verify solution with forward kinematics
        if (!verifySolution(solution, target_frame)) {
            RCLCPP_WARN(nh_.get_logger(), "IK solution verification failed");
            return false;
        }
        
        RCLCPP_DEBUG(nh_.get_logger(), 
                    "IK solved in %.3f ms using %s method (avg: %.3f ms)",
                    elapsed_ms, method_name.c_str(), total_time_ / total_calls_);
        
        return true;
    } else {
        RCLCPP_WARN(nh_.get_logger(), 
                   "IK failed with error %d using %s method (took %.3f ms)",
                   ik_result, method_name.c_str(), elapsed_ms);
        return false;
    }
}

int InverseKinematics::solveIKIterative(const KDL::Frame& target_frame,
                                        const KDL::JntArray& seed,
                                        KDL::JntArray& solution) {
    // Implement iterative IK with gradient descent
    const int max_iterations = 500;
    const double tolerance = 1e-6;
    const double alpha = 0.1;  // Learning rate
    
    solution = seed;
    KDL::Frame current_frame;
    KDL::Twist error;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute current forward kinematics
        if (fk_solver_->JntToCart(solution, current_frame) < 0) {
            return -1;
        }
        
        // Compute error twist
        error = KDL::diff(current_frame, target_frame);
        
        // Check convergence
        if (error.vel.Norm() < tolerance && error.rot.Norm() < tolerance) {
            return iter;  // Return number of iterations
        }
        
        // Compute Jacobian
        KDL::Jacobian jacobian(num_joints_);
        // ... Jacobian computation implementation
        
        // Update joint angles using pseudo-inverse
        // ... Implementation of gradient descent update
    }
    
    return -1;  // Max iterations reached
}

bool InverseKinematics::checkJointLimits(const std::vector<double>& joint_positions) const {
    if (joint_positions.size() != num_joints_) {
        RCLCPP_ERROR(nh_.get_logger(), 
                    "Joint positions size mismatch: expected %zu, got %zu",
                    num_joints_, joint_positions.size());
        return false;
    }
    
    for (size_t i = 0; i < num_joints_; ++i) {
        if (joint_positions[i] < joint_lower_limits_[i] || 
            joint_positions[i] > joint_upper_limits_[i]) {
            RCLCPP_DEBUG(nh_.get_logger(), 
                        "Joint %s out of bounds: %.3f not in [%.3f, %.3f]",
                        joint_names_[i].c_str(),
                        joint_positions[i],
                        joint_lower_limits_[i],
                        joint_upper_limits_[i]);
            return false;
        }
    }
    
    return true;
}

bool InverseKinematics::verifySolution(const std::vector<double>& joint_positions,
                                       const KDL::Frame& target_frame) const {
    KDL::JntArray joint_array(num_joints_);
    for (size_t i = 0; i < num_joints_; ++i) {
        joint_array(i) = joint_positions[i];
    }
    
    KDL::Frame computed_frame;
    if (fk_solver_->JntToCart(joint_array, computed_frame) < 0) {
        return false;
    }
    
    // Compute position error
    double position_error = KDL::diff(computed_frame.p, target_frame.p).Norm();
    double orientation_error = KDL::diff(computed_frame.M, target_frame.M).Norm();
    
    const double position_tolerance = 0.001;  // 1 mm
    const double orientation_tolerance = 0.01;  // ~0.57 degrees
    
    if (position_error > position_tolerance || orientation_error > orientation_tolerance) {
        RCLCPP_DEBUG(nh_.get_logger(), 
                    "Verification failed: pos error=%.6f, orient error=%.6f",
                    position_error, orientation_error);
        return false;
    }
    
    return true;
}

std::vector<double> InverseKinematics::getJointNames() const {
    return joint_names_;
}

std::vector<double> InverseKinematics::getJointLimitsLower() const {
    return joint_lower_limits_;
}

std::vector<double> InverseKinematics::getJointLimitsUpper() const {
    return joint_upper_limits_;
}

double InverseKinematics::getAverageSolveTime() const {
    return total_calls_ > 0 ? total_time_ / total_calls_ : 0.0;
}

uint64_t InverseKinematics::getTotalCalls() const {
    return total_calls_;
}

}  // namespace motion_planning
