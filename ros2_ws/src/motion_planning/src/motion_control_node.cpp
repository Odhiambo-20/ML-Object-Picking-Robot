/**
 * @file inverse_kinematics.cpp
 * @brief Advanced Inverse Kinematics solver using KDL with multiple resolution strategies
 * 
 * This module implements a comprehensive inverse kinematics solver using the
 * Kinematics and Dynamics Library (KDL) with support for multiple solution
 * strategies, redundancy resolution, and singularity handling.
 * 
 * @note Production-grade code for industrial robotic manipulators with
 * comprehensive error handling and performance optimization.
 */

#include "motion_planning/inverse_kinematics.hpp"
#include <rclcpp/rclcpp.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/utilities/error.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <chrono>
#include <functional>
#include <limits>

namespace motion_planning
{

class InverseKinematics::Impl
{
public:
    Impl() = delete;
    
    Impl(const std::shared_ptr<moveit::core::RobotModel>& robot_model,
         const rclcpp::Node::SharedPtr& node,
         const std::string& group_name)
        : robot_model_(robot_model)
        , node_(node)
        , group_name_(group_name)
        , chain_()
        , fk_solver_(nullptr)
        , ik_solver_vel_(nullptr)
        , ik_solver_pos_(nullptr)
        , jacobian_solver_(nullptr)
        , ik_solver_type_(IKSolverType::NR_JL)
        , max_iterations_(500)
        , tolerance_(1e-6)
        , lambda_(0.01)
        , singularity_threshold_(1e-3)
        , joint_velocity_limits_()
        , joint_acceleration_limits_()
        , redundancy_resolution_enabled_(true)
        , singularity_avoidance_enabled_(true)
        , joint_limit_avoidance_enabled_(true)
        , collision_avoidance_weight_(0.1)
        , manipulability_weight_(0.2)
        , joint_limit_weight_(0.3)
        , energy_weight_(0.1)
    {
        initializeFromRobotModel();
        loadConfigurationParameters();
        setupIKSolvers();
        
        RCLCPP_INFO(node_->get_logger(),
                   "InverseKinematics initialized for group: %s with %zu DOF",
                   group_name_.c_str(), chain_.getNrOfJoints());
    }
    
    ~Impl() = default;
    
    /**
     * @brief Solve inverse kinematics for a target pose
     * @param target_pose Desired end-effector pose
     * @param initial_guess Initial joint configuration (optional)
     * @param constraints Additional constraints (joint limits, obstacles, etc.)
     * @return IKSolution structure with results
     */
    IKSolution solveIK(const geometry_msgs::msg::Pose& target_pose,
                      const std::vector<double>& initial_guess,
                      const IKConstraints& constraints)
    {
        IKSolution solution;
        solution.success = false;
        solution.iterations = 0;
        solution.error_norm = std::numeric_limits<double>::max();
        solution.singularity_detected = false;
        solution.computation_time_ms = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        try
        {
            // Convert target pose to KDL Frame
            KDL::Frame target_frame = poseToKDLFrame(target_pose);
            
            // Prepare joint array
            KDL::JntArray q_init(chain_.getNrOfJoints());
            KDL::JntArray q_result(chain_.getNrOfJoints());
            
            // Set initial guess or use current state
            if (!initial_guess.empty() && initial_guess.size() == chain_.getNrOfJoints())
            {
                for (size_t i = 0; i < chain_.getNrOfJoints(); ++i)
                {
                    q_init(i) = initial_guess[i];
                }
            }
            else
            {
                // Use a neutral configuration
                for (size_t i = 0; i < chain_.getNrOfJoints(); ++i)
                {
                    q_init(i) = 0.0;
                }
            }
            
            // Apply constraints to initial guess
            applyConstraintsToInitialGuess(q_init, constraints);
            
            // Solve IK based on selected solver type
            int ik_result = -1;
            KDL::JntArray q_out(chain_.getNrOfJoints());
            
            switch (ik_solver_type_)
            {
                case IKSolverType::NR_JL:
                    ik_result = solveIKNRJL(target_frame, q_init, q_out, constraints, solution);
                    break;
                    
                case IKSolverType::LMA:
                    ik_result = solveIKLMA(target_frame, q_init, q_out, constraints, solution);
                    break;
                    
                case IKSolverType::VEL_PINV:
                    ik_result = solveIKVelPinv(target_frame, q_init, q_out, constraints, solution);
                    break;
                    
                case IKSolverType::VEL_WDLS:
                    ik_result = solveIKVelWDLS(target_frame, q_init, q_out, constraints, solution);
                    break;
                    
                case IKSolverType::OPTIMIZATION:
                    ik_result = solveIKOptimization(target_frame, q_init, q_out, constraints, solution);
                    break;
                    
                default:
                    RCLCPP_ERROR(node_->get_logger(), "Unknown IK solver type");
                    break;
            }
            
            if (ik_result >= 0)
            {
                solution.success = true;
                
                // Convert KDL joints to vector
                solution.joint_positions.resize(chain_.getNrOfJoints());
                for (size_t i = 0; i < chain_.getNrOfJoints(); ++i)
                {
                    solution.joint_positions[i] = q_out(i);
                }
                
                // Validate solution
                validateSolution(q_out, target_frame, constraints, solution);
                
                // Apply redundancy resolution if needed
                if (redundancy_resolution_enabled_ && chain_.getNrOfJoints() > 6)
                {
                    applyRedundancyResolution(q_out, target_frame, constraints, solution);
                }
                
                // Compute manipulability measure
                computeManipulability(q_out, solution);
                
                // Check for singularities
                checkForSingularities(q_out, solution);
                
                solution_stats_.successful_solutions++;
            }
            else
            {
                RCLCPP_DEBUG(node_->get_logger(), 
                           "IK solver failed with code: %d", ik_result);
                solution.success = false;
                solution_stats_.failed_solutions++;
                
                // Attempt fallback solver
                if (attemptFallbackSolution(target_frame, q_init, q_out, constraints, solution))
                {
                    solution.success = true;
                    solution_stats_.fallback_solutions++;
                }
            }
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(node_->get_logger(),
                       "Exception in IK solver: %s", e.what());
            solution.success = false;
            solution.error_message = std::string("IK solver exception: ") + e.what();
            solution_stats_.exceptions++;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        solution.computation_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        solution_stats_.total_calls++;
        solution_stats_.total_computation_time_ms += solution.computation_time_ms;
        solution_stats_.average_computation_time_ms = 
            solution_stats_.total_computation_time_ms / solution_stats_.total_calls;
        
        // Log performance if computation took too long
        if (solution.computation_time_ms > 100)  // 100ms threshold
        {
            RCLCPP_WARN(node_->get_logger(),
                       "IK computation took %ldms (iterations: %d)",
                       solution.computation_time_ms, solution.iterations);
        }
        
        return solution;
    }
    
    /**
     * @brief Solve IK for multiple target poses (trajectory)
     * @param target_poses Vector of target poses
     * @param initial_guess Initial joint configuration
     * @param constraints IK constraints
     * @param continuity_weight Weight for solution continuity
     * @return Vector of IK solutions
     */
    std::vector<IKSolution> solveIKTrajectory(
        const std::vector<geometry_msgs::msg::Pose>& target_poses,
        const std::vector<double>& initial_guess,
        const IKConstraints& constraints,
        double continuity_weight = 0.5)
    {
        std::vector<IKSolution> solutions;
        solutions.reserve(target_poses.size());
        
        if (target_poses.empty())
        {
            RCLCPP_WARN(node_->get_logger(), "Empty trajectory provided for IK");
            return solutions;
        }
        
        std::vector<double> current_guess = initial_guess;
        
        for (size_t i = 0; i < target_poses.size(); ++i)
        {
            // Adjust constraints for trajectory point
            IKConstraints point_constraints = constraints;
            
            // Add continuity constraint
            if (i > 0 && continuity_weight > 0.0)
            {
                point_constraints.continuity_constraint = true;
                point_constraints.previous_solution = solutions[i-1].joint_positions;
                point_constraints.continuity_weight = continuity_weight;
            }
            
            // Solve IK for this point
            IKSolution solution = solveIK(target_poses[i], current_guess, point_constraints);
            
            if (solution.success)
            {
                // Update guess for next point
                current_guess = solution.joint_positions;
            }
            else
            {
                // Attempt recovery if IK fails
                RCLCPP_WARN(node_->get_logger(),
                           "IK failed for trajectory point %zu, attempting recovery", i);
                
                if (attemptTrajectoryRecovery(i, target_poses, solutions, constraints, solution))
                {
                    current_guess = solution.joint_positions;
                }
                else
                {
                    // Mark entire trajectory as failed
                    RCLCPP_ERROR(node_->get_logger(),
                               "Trajectory IK failed at point %zu, aborting", i);
                    break;
                }
            }
            
            solutions.push_back(solution);
        }
        
        // Smooth trajectory solutions
        if (constraints.smooth_trajectory)
        {
            smoothTrajectorySolutions(solutions);
        }
        
        return solutions;
    }
    
    /**
     * @brief Compute Jacobian matrix at given joint configuration
     * @param joint_positions Joint configuration
     * @param reference_point Reference point for Jacobian (default: end-effector)
     * @return Jacobian matrix (6 x n)
     */
    Eigen::MatrixXd computeJacobian(const std::vector<double>& joint_positions,
                                   const Eigen::Vector3d& reference_point = Eigen::Vector3d::Zero())
    {
        if (joint_positions.size() != chain_.getNrOfJoints())
        {
            throw std::invalid_argument("Joint position size mismatch");
        }
        
        KDL::JntArray q(chain_.getNrOfJoints());
        for (size_t i = 0; i < chain_.getNrOfJoints(); ++i)
        {
            q(i) = joint_positions[i];
        }
        
        KDL::Jacobian jacobian(chain_.getNrOfJoints());
        KDL::Frame frame;
        
        if (jacobian_solver_)
        {
            int result = jacobian_solver_->JntToJac(q, jacobian);
            
            if (result >= 0)
            {
                // Convert KDL Jacobian to Eigen
                Eigen::MatrixXd eigen_jacobian(6, chain_.getNrOfJoints());
                for (int i = 0; i < 6; ++i)
                {
                    for (unsigned int j = 0; j < chain_.getNrOfJoints(); ++j)
                    {
                        eigen_jacobian(i, j) = jacobian(i, j);
                    }
                }
                
                // Adjust for reference point if needed
                if (!reference_point.isZero())
                {
                    adjustJacobianForReferencePoint(q, reference_point, eigen_jacobian);
                }
                
                return eigen_jacobian;
            }
        }
        
        throw std::runtime_error("Failed to compute Jacobian");
    }
    
    /**
     * @brief Compute manipulability measure at given configuration
     * @param joint_positions Joint configuration
     * @return Manipulability measure (scalar)
     */
    double computeManipulabilityMeasure(const std::vector<double>& joint_positions)
    {
        try
        {
            Eigen::MatrixXd J = computeJacobian(joint_positions);
            Eigen::MatrixXd JJT = J * J.transpose();
            
            // Compute manipulability as sqrt(det(J*J^T))
            double det = JJT.determinant();
            if (det > 0)
            {
                return std::sqrt(det);
            }
            return 0.0;
        }
        catch (...)
        {
            return 0.0;
        }
    }
    
    /**
     * @brief Check for kinematic singularities
     * @param joint_positions Joint configuration to check
     * @return SingularityReport with analysis results
     */
    SingularityReport checkSingularities(const std::vector<double>& joint_positions)
    {
        SingularityReport report;
        report.is_singular = false;
        report.condition_number = 0.0;
        report.manipulability = 0.0;
        report.singularity_type = SingularityType::NONE;
        
        try
        {
            Eigen::MatrixXd J = computeJacobian(joint_positions);
            
            // Compute condition number
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(J);
            Eigen::VectorXd singular_values = svd.singularValues();
            
            double min_sv = singular_values.minCoeff();
            double max_sv = singular_values.maxCoeff();
            
            report.condition_number = (min_sv > 0) ? (max_sv / min_sv) : std::numeric_limits<double>::infinity();
            report.manipulability = computeManipulabilityMeasure(joint_positions);
            
            // Check for singularity
            if (min_sv < singularity_threshold_)
            {
                report.is_singular = true;
                
                // Determine singularity type based on rank deficiency
                int rank_deficiency = 0;
                for (int i = 0; i < singular_values.size(); ++i)
                {
                    if (singular_values[i] < singularity_threshold_)
                    {
                        rank_deficiency++;
                    }
                }
                
                if (rank_deficiency == 1)
                {
                    report.singularity_type = SingularityType::WRIST;
                }
                else if (rank_deficiency == 2)
                {
                    report.singularity_type = SingularityType::ELBOW;
                }
                else if (rank_deficiency >= 3)
                {
                    report.singularity_type = SingularityType::SHOULDER;
                }
            }
            
            // Store singular vectors for singularity escape
            if (report.is_singular)
            {
                report.null_space_basis = svd.matrixV().rightCols(rank_deficiency);
            }
        }
        catch (const std::exception& e)
        {
            RCLCPP_ERROR(node_->get_logger(),
                       "Failed to check singularities: %s", e.what());
        }
        
        return report;
    }
    
    /**
     * @brief Set IK solver type
     * @param solver_type Type of IK solver to use
     */
    void setIKSolverType(IKSolverType solver_type)
    {
        ik_solver_type_ = solver_type;
        setupIKSolvers();
        
        RCLCPP_INFO(node_->get_logger(),
                   "IK solver type set to: %d", static_cast<int>(solver_type));
    }
    
    /**
     * @brief Set solver parameters
     * @param max_iterations Maximum number of iterations
     * @param tolerance Convergence tolerance
     * @param lambda Damping factor for numerical methods
     */
    void setSolverParameters(int max_iterations, double tolerance, double lambda)
    {
        max_iterations_ = max_iterations;
        tolerance_ = tolerance;
        lambda_ = lambda;
        
        RCLCPP_DEBUG(node_->get_logger(),
                    "IK solver parameters updated: max_iter=%d, tol=%.2e, lambda=%.3f",
                    max_iterations_, tolerance_, lambda_);
    }
    
    /**
     * @brief Enable/disable redundancy resolution
     * @param enable True to enable redundancy resolution
     */
    void enableRedundancyResolution(bool enable)
    {
        redundancy_resolution_enabled_ = enable;
        RCLCPP_INFO(node_->get_logger(),
                   "Redundancy resolution %s",
                   enable ? "enabled" : "disabled");
    }
    
    /**
     * @brief Set redundancy resolution weights
     * @param weights Vector of weights for different criteria
     */
    void setRedundancyWeights(const RedundancyWeights& weights)
    {
        collision_avoidance_weight_ = weights.collision_avoidance;
        manipulability_weight_ = weights.manipulability;
        joint_limit_weight_ = weights.joint_limit;
        energy_weight_ = weights.energy;
        
        RCLCPP_DEBUG(node_->get_logger(),
                    "Redundancy weights updated: coll=%.2f, manip=%.2f, jlim=%.2f, energy=%.2f",
                    collision_avoidance_weight_, manipulability_weight_,
                    joint_limit_weight_, energy_weight_);
    }
    
    /**
     * @brief Get IK solver statistics
     * @return IKSolverStats structure with statistics
     */
    IKSolverStats getStatistics() const
    {
        return solution_stats_;
    }
    
    /**
     * @brief Reset solver statistics
     */
    void resetStatistics()
    {
        solution_stats_ = IKSolverStats();
        RCLCPP_INFO(node_->get_logger(), "IK solver statistics reset");
    }

private:
    /**
     * @brief Initialize KDL chain from robot model
     */
    void initializeFromRobotModel()
    {
        const moveit::core::JointModelGroup* jmg = robot_model_->getJointModelGroup(group_name_);
        if (!jmg)
        {
            throw std::runtime_error("Joint model group not found: " + group_name_);
        }
        
        // Get kinematic chain from robot model
        const std::vector<const moveit::core::LinkModel*>& link_models = jmg->getLinkModels();
        const std::vector<const moveit::core::JointModel*>& joint_models = jmg->getJointModels();
        
        // Create KDL chain
        for (size_t i = 0; i < joint_models.size(); ++i)
        {
            const moveit::core::JointModel* joint_model = joint_models[i];
            
            if (joint_model->getType() == moveit::core::JointModel::REVOLUTE ||
                joint_model->getType() == moveit::core::JointModel::PRISMATIC)
            {
                // Get joint axis and origin
                Eigen::Vector3d axis = joint_model->getAxis();
                Eigen::Isometry3d origin = joint_model->getChildLinkModel()->getJointOriginTransform();
                
                // Convert to KDL
                KDL::Vector kdl_origin(origin.translation().x(),
                                      origin.translation().y(),
                                      origin.translation().z());
                KDL::Vector kdl_axis(axis.x(), axis.y(), axis.z());
                
                // Create KDL joint
                KDL::Joint::JointType kdl_joint_type;
                if (joint_model->getType() == moveit::core::JointModel::REVOLUTE)
                {
                    kdl_joint_type = KDL::Joint::RotAxis;
                }
                else
                {
                    kdl_joint_type = KDL::Joint::TransAxis;
                }
                
                KDL::Joint kdl_joint(joint_model->getName(), kdl_origin, kdl_axis, kdl_joint_type);
                
                // Create KDL segment
                KDL::Frame kdl_frame(KDL::Rotation::Quaternion(
                    Eigen::Quaterniond(origin.rotation()).x(),
                    Eigen::Quaterniond(origin.rotation()).y(),
                    Eigen::Quaterniond(origin.rotation()).z(),
                    Eigen::Quaterniond(origin.rotation()).w()),
                    kdl_origin);
                
                KDL::Segment kdl_segment(joint_model->getName(), kdl_joint, kdl_frame);
                chain_.addSegment(kdl_segment);
            }
        }
        
        // Store joint limits
        const std::vector<std::pair<double, double>>& joint_limits = jmg->getVariableBoundsPairs();
        for (const auto& limit : joint_limits)
        {
            joint_position_limits_.push_back(limit);
        }
        
        // Initialize velocity and acceleration limits (default values)
        for (size_t i = 0; i < chain_.getNrOfJoints(); ++i)
        {
            joint_velocity_limits_.push_back(2.0);  // rad/s or m/s
            joint_acceleration_limits_.push_back(5.0);  // rad/s² or m/s²
        }
    }
    
    /**
     * @brief Load configuration parameters from ROS
     */
    void loadConfigurationParameters()
    {
        node_->declare_parameter<int>("ik_max_iterations", 500);
        node_->declare_parameter<double>("ik_tolerance", 1e-6);
        node_->declare_parameter<double>("ik_lambda", 0.01);
        node_->declare_parameter<double>("singularity_threshold", 1e-3);
        node_->declare_parameter<bool>("redundancy_resolution", true);
        node_->declare_parameter<bool>("singularity_avoidance", true);
        node_->declare_parameter<bool>("joint_limit_avoidance", true);
        
        max_iterations_ = node_->get_parameter("ik_max_iterations").as_int();
        tolerance_ = node_->get_parameter("ik_tolerance").as_double();
        lambda_ = node_->get_parameter("ik_lambda").as_double();
        singularity_threshold_ = node_->get_parameter("singularity_threshold").as_double();
        redundancy_resolution_enabled_ = node_->get_parameter("redundancy_resolution").as_bool();
        singularity_avoidance_enabled_ = node_->get_parameter("singularity_avoidance").as_bool();
        joint_limit_avoidance_enabled_ = node_->get_parameter("joint_limit_avoidance").as_bool();
    }
    
    /**
     * @brief Set up IK solvers based on selected type
     */
    void setupIKSolvers()
    {
        // Create FK solver
        fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
        
        // Create Jacobian solver
        jacobian_solver_ = std::make_unique<KDL::ChainJntToJacSolver>(chain_);
        
        // Create IK solvers based on type
        switch (ik_solver_type_)
        {
            case IKSolverType::NR_JL:
                setupNRJLSolver();
                break;
                
            case IKSolverType::LMA:
                setupLMASolver();
                break;
                
            case IKSolverType::VEL_PINV:
                setupVelPinvSolver();
                break;
                
            case IKSolverType::VEL_WDLS:
                setupVelWDLSSolver();
                break;
                
            case IKSolverType::OPTIMIZATION:
                // Optimization solver doesn't need pre-setup
                break;
                
            default:
                RCLCPP_WARN(node_->get_logger(),
                           "Unknown solver type, defaulting to NR_JL");
                ik_solver_type_ = IKSolverType::NR_JL;
                setupNRJLSolver();
                break;
        }
    }
    
    /**
     * @brief Set up Newton-Raphson with joint limits solver
     */
    void setupNRJLSolver()
    {
        // Create joint limit arrays
        KDL::JntArray q_min(chain_.getNrOfJoints());
        KDL::JntArray q_max(chain_.getNrOfJoints());
        
        for (size_t i = 0; i < chain_.getNrOfJoints(); ++i)
        {
            q_min(i) = joint_position_limits_[i].first;
            q_max(i) = joint_position_limits_[i].second;
        }
        
        // Create velocity IK solver for internal use
        auto ik_solver_vel = std::make_unique<KDL::ChainIkSolverVel_pinv>(chain_);
        
        // Create position IK solver
        ik_solver_pos_ = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(
            chain_, q_min, q_max, *fk_solver_, *ik_solver_vel,
            max_iterations_, tolerance_);
    }
    
    /**
     * @brief Set up Levenberg-Marquardt solver
     */
    void setupLMASolver()
    {
        // Create joint limit arrays
        KDL::JntArray q_min(chain_.getNrOfJoints());
        KDL::JntArray q_max(chain_.getNrOfJoints());
        
        for (size_t i = 0; i < chain_.getNrOfJoints(); ++i)
        {
            q_min(i) = joint_position_limits_[i].first;
            q_max(i) = joint_position_limits_[i].second;
        }
        
        // Create LMA solver
        ik_solver_pos_ = std::make_unique<KDL::ChainIkSolverPos_LMA>(
            chain_, q_min, q_max, max_iterations_, tolerance_);
    }
    
    /**
     * @brief Set up velocity-level pseudo-inverse solver
     */
    void setupVelPinvSolver()
    {
        ik_solver_vel_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(chain_);
    }
    
    /**
     * @brief Set up velocity-level Weighted Damped Least Squares solver
     */
    void setupVelWDLSSolver()
    {
        ik_solver_vel_ = std::make_unique<KDL::ChainIkSolverVel_wdls>(chain_);
        
        // Set damping and weights
        KDL::ChainIkSolverVel_wdls* wdls_solver = 
            dynamic_cast<KDL::ChainIkSolverVel_wdls*>(ik_solver_vel_.get());
        
        if (wdls_solver)
        {
            wdls_solver->setLambda(lambda_);
            
            // Set weights for Cartesian error (position and orientation)
            Eigen::MatrixXd weights = Eigen::MatrixXd::Identity(6, 6);
            weights.block<3,3>(0,0) *= 1.0;  // Position weights
            weights.block<3,3>(3,3) *= 0.5;  // Orientation weights
            
            wdls_solver->setWeightTS(weights);
        }
    }
    
    /**
     * @brief Solve IK using Newton-Raphson with joint limits
     */
    int solveIKNRJL(const KDL::Frame& target_frame,
                   const KDL::JntArray& q_init,
                   KDL::JntArray& q_out,
                   const IKConstraints& constraints,
                   IKSolution& solution)
    {
        if (!ik_solver_pos_)
        {
            return -1;
        }
        
        int result = ik_solver_pos_->CartToJnt(q_init, target_frame, q_out);
        solution.iterations = max_iterations_;  // NR doesn't return iteration count
        
        // Compute error
        KDL::Frame achieved_frame;
        fk_solver_->JntToCart(q_out, achieved_frame);
        solution.error_norm = computeFrameError(target_frame, achieved_frame);
        
        return result;
    }
    
    /**
     * @brief Solve IK using Levenberg-Marquardt algorithm
     */
    int solveIKLMA(const KDL::Frame& target_frame,
                  const KDL::JntArray& q_init,
                  KDL::JntArray& q_out,
                  const IKConstraints& constraints,
                  IKSolution& solution)
    {
        if (!ik_solver_pos_)
        {
            return -1;
        }
        
        int result = ik_solver_pos_->CartToJnt(q_init, target_frame, q_out);
        
        // LMA solver doesn't provide iteration count or error directly
        // Compute error manually
        KDL::Frame achieved_frame;
        fk_solver_->JntToCart(q_out, achieved_frame);
        solution.error_norm = computeFrameError(target_frame, achieved_frame);
        
        return result;
    }
    
    /**
     * @brief Solve IK using velocity-level pseudo-inverse
     */
    int solveIKVelPinv(const KDL::Frame& target_frame,
                      const KDL::JntArray& q_init,
                      KDL::JntArray& q_out,
                      const IKConstraints& constraints,
                      IKSolution& solution)
    {
        if (!ik_solver_vel_ || !fk_solver_)
        {
            return -1;
        }
        
        KDL::JntArray q_current = q_init;
        KDL::Frame current_frame;
        KDL::Twist desired_twist;
        
        for (int iter = 0; iter < max_iterations_; ++iter)
        {
            // Compute current forward kinematics
            fk_solver_->JntToCart(q_current, current_frame);
            
            // Compute error twist
            desired_twist = diffRelative(target_frame, current_frame);
            
            // Check convergence
            double error = desired_twist.vel.Norm() + desired_twist.rot.Norm();
            if (error < tolerance_)
            {
                q_out = q_current;
                solution.iterations = iter;
                solution.error_norm = error;
                return 0;
            }
            
            // Compute joint velocities
            KDL::JntArray qdot(chain_.getNrOfJoints());
            int vel_result = ik_solver_vel_->CartToJnt(q_current, desired_twist, qdot);
            
            if (vel_result < 0)
            {
                return vel_result;
            }
            
            // Update joint positions
            for (unsigned int i = 0; i < chain_.getNrOfJoints(); ++i)
            {
                q_current(i) += qdot(i) * 0.1;  // Step size
                
                // Apply joint limits
                q_current(i) = std::max(joint_position_limits_[i].first,
                                       std::min(joint_position_limits_[i].second, q_current(i)));
            }
        }
        
        return -2;  // Max iterations exceeded
    }
    
    /**
     * @brief Solve IK using velocity-level WDLS
     */
    int solveIKVelWDLS(const KDL::Frame& target_frame,
                      const KDL::JntArray& q_init,
                      KDL::JntArray& q_out,
                      const IKConstraints& constraints,
                      IKSolution& solution)
    {
        // Similar to pseudo-inverse but with WDLS solver
        return solveIKVelPinv(target_frame, q_init, q_out, constraints, solution);
    }
    
    /**
     * @brief Solve IK using optimization-based approach
     */
    int solveIKOptimization(const KDL::Frame& target_frame,
                           const KDL::JntArray& q_init,
                           KDL::JntArray& q_out,
                           const IKConstraints& constraints,
                           IKSolution& solution)
    {
        // This is a simplified implementation. In production, you would use
        // a proper optimization library like NLopt or Ceres Solver.
        
        const int n = chain_.getNrOfJoints();
        Eigen::VectorXd q = Eigen::VectorXd::Zero(n);
        for (int i = 0; i < n; ++i)
        {
            q(i) = q_init(i);
        }
        
        // Define cost function
        auto cost_function = [&](const Eigen::VectorXd& x) -> double {
            KDL::JntArray q_test(n);
            for (int i = 0; i < n; ++i)
            {
                q_test(i) = x(i);
            }
            
            KDL::Frame frame;
            fk_solver_->JntToCart(q_test, frame);
            
            double position_error = (frame.p - target_frame.p).Norm();
            double orientation_error = 0.5 * (1.0 - (frame.M.UnitX() * target_frame.M.UnitX() +
                                                   frame.M.UnitY() * target_frame.M.UnitY() +
                                                   frame.M.UnitZ() * target_frame.M.UnitZ()));
            
            double cost = position_error * position_error + orientation_error * orientation_error;
            
            // Add constraints to cost
            if (joint_limit_avoidance_enabled_)
            {
                cost += computeJointLimitCost(x);
            }
            
            if (constraints.collision_avoidance_enabled)
            {
                cost += computeCollisionAvoidanceCost(x, constraints.collision_data);
            }
            
            return cost;
        };
        
        // Simple gradient descent (replace with proper optimizer in production)
        double learning_rate = 0.01;
        double prev_cost = cost_function(q);
        
        for (int iter = 0; iter < max_iterations_; ++iter)
        {
            // Compute numerical gradient
            Eigen::VectorXd gradient = Eigen::VectorXd::Zero(n);
            for (int i = 0; i < n; ++i)
            {
                Eigen::VectorXd q_plus = q;
                Eigen::VectorXd q_minus = q;
                
                double epsilon = 1e-6;
                q_plus(i) += epsilon;
                q_minus(i) -= epsilon;
                
                gradient(i) = (cost_function(q_plus) - cost_function(q_minus)) / (2.0 * epsilon);
            }
            
            // Update
            q -= learning_rate * gradient;
            
            // Apply joint limits
            for (int i = 0; i < n; ++i)
            {
                q(i) = std::max(joint_position_limits_[i].first,
                               std::min(joint_position_limits_[i].second, q(i)));
            }
            
            // Check convergence
            double current_cost = cost_function(q);
            if (std::abs(prev_cost - current_cost) < tolerance_)
            {
                break;
            }
            prev_cost = current_cost;
        }
        
        // Copy result
        for (int i = 0; i < n; ++i)
        {
            q_out(i) = q(i);
        }
        
        // Compute final error
        KDL::Frame achieved_frame;
        fk_solver_->JntToCart(q_out, achieved_frame);
        solution.error_norm = computeFrameError(target_frame, achieved_frame);
        solution.iterations = max_iterations_;
        
        return 0;
    }
    
    /**
     * @brief Attempt fallback solution when primary solver fails
     */
    bool attemptFallbackSolution(const KDL::Frame& target_frame,
                                const KDL::JntArray& q_init,
                                KDL::JntArray& q_out,
                                const IKConstraints& constraints,
                                IKSolution& solution)
    {
        RCLCPP_WARN(node_->get_logger(), "Attempting fallback IK solution");
        
        // Try different solver types in order of preference
        std::vector<IKSolverType> fallback_order = {
            IKSolverType::LMA,
            IKSolverType::OPTIMIZATION,
            IKSolverType::VEL_WDLS,
            IKSolverType::VEL_PINV
        };
        
        for (const auto& solver_type : fallback_order)
        {
            if (solver_type == ik_solver_type_)
            {
                continue;  // Skip current solver
            }
            
            // Temporarily switch solver
            IKSolverType original_type = ik_solver_type_;
            ik_solver_type_ = solver_type;
            setupIKSolvers();
            
            IKSolution fallback_solution;
            int result = -1;
            
            switch (solver_type)
            {
                case IKSolverType::LMA:
                    result = solveIKLMA(target_frame, q_init, q_out, constraints, fallback_solution);
                    break;
                case IKSolverType::OPTIMIZATION:
                    result = solveIKOptimization(target_frame, q_init, q_out, constraints, fallback_solution);
                    break;
                case IKSolverType::VEL_WDLS:
                    result = solveIKVelWDLS(target_frame, q_init, q_out, constraints, fallback_solution);
                    break;
                case IKSolverType::VEL_PINV:
                    result = solveIKVelPinv(target_frame, q_init, q_out, constraints, fallback_solution);
                    break;
                default:
                    break;
            }
            
            // Restore original solver
            ik_solver_type_ = original_type;
            setupIKSolvers();
            
            if (result >= 0)
            {
                solution = fallback_solution;
                RCLCPP_INFO(node_->get_logger(),
                          "Fallback IK succeeded with solver type: %d",
                          static_cast<int>(solver_type));
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * @brief Attempt trajectory recovery when IK fails
     */
    bool attemptTrajectoryRecovery(size_t failed_index,
                                  const std::vector<geometry_msgs::msg::Pose>& target_poses,
                                  std::vector<IKSolution>& solutions,
                                  const IKConstraints& constraints,
                                  IKSolution& recovered_solution)
    {
        // Strategy 1: Interpolate between neighboring successful solutions
        if (failed_index > 0 && failed_index < target_poses.size() - 1)
        {
            if (solutions[failed_index - 1].success)
            {
                // Use previous successful solution as initial guess
                IKConstraints relaxed_constraints = constraints;
                relaxed_constraints.position_tolerance *= 2.0;
                relaxed_constraints.orientation_tolerance *= 2.0;
                
                recovered_solution = solveIK(target_poses[failed_index],
                                           solutions[failed_index - 1].joint_positions,
                                           relaxed_constraints);
                
                if (recovered_solution.success)
                {
                    return true;
                }
            }
        }
        
        // Strategy 2: Use a different IK solver
        IKSolverType original_type = ik_solver_type_;
        setIKSolverType(IKSolverType::LMA);  // LMA is often more robust
        
        recovered_solution = solveIK(target_poses[failed_index],
                                   std::vector<double>(),
                                   constraints);
        
        setIKSolverType(original_type);  // Restore original solver
        
        return recovered_solution.success;
    }
    
    /**
     * @brief Apply constraints to initial guess
     */
    void applyConstraintsToInitialGuess(KDL::JntArray& q_init,
                                       const IKConstraints& constraints)
    {
        // Apply joint limits
        for (size_t i = 0; i < chain_.getNrOfJoints(); ++i)
        {
            q_init(i) = std::max(joint_position_limits_[i].first,
                                std::min(joint_position_limits_[i].second, q_init(i)));
        }
        
        // Apply preferred joint values if specified
        if (!constraints.preferred_joint_values.empty() &&
            constraints.preferred_joint_values.size() == chain_.getNrOfJoints())
        {
            for (size_t i = 0; i < chain_.getNrOfJoints(); ++i)
            {
                double weight = constraints.preferred_joint_weights.empty() ? 0.5 :
                               constraints.preferred_joint_weights[i];
                q_init(i) = q_init(i) * (1.0 - weight) +
                           constraints.preferred_joint_values[i] * weight;
            }
        }
    }
    
    /**
     * @brief Validate IK solution
     */
    void validateSolution(const KDL::JntArray& q,
                         const KDL::Frame& target_frame,
                         const IKConstraints& constraints,
                         IKSolution& solution)
    {
        // Check joint limits
        for (size_t i = 0; i < chain_.getNrOfJoints(); ++i)
        {
            if (q(i) < joint_position_limits_[i].first ||
                q(i) > joint_position_limits_[i].second)
            {
                solution.valid = false;
                solution.validation_errors.push_back(
                    "Joint " + std::to_string(i) + " exceeds limits");
                break;
            }
        }
        
        // Check position error
        KDL::Frame achieved_frame;
        fk_solver_->JntToCart(q, achieved_frame);
        
        double position_error = (achieved_frame.p - target_frame.p).Norm();
        if (position_error > constraints.position_tolerance)
        {
            solution.valid = false;
            solution.validation_errors.push_back(
                "Position error too large: " + std::to_string(position_error));
        }
        
        // Check orientation error
        double orientation_error = computeOrientationError(achieved_frame.M, target_frame.M);
        if (orientation_error > constraints.orientation_tolerance)
        {
            solution.valid = false;
            solution.validation_errors.push_back(
                "Orientation error too large: " + std::to_string(orientation_error));
        }
        
        if (solution.valid)
        {
            solution_stats_.valid_solutions++;
        }
        else
        {
            solution_stats_.invalid_solutions++;
        }
    }
    
    /**
     * @brief Apply redundancy resolution
     */
    void applyRedundancyResolution(KDL::JntArray& q,
                                  const KDL::Frame& target_frame,
                                  const IKConstraints& constraints,
                                  IKSolution& solution)
    {
        if (chain_.getNrOfJoints() <= 6)
        {
            return;  // Not a redundant manipulator
        }
        
        // Compute null space projection
        Eigen::MatrixXd J = computeJacobian(solution.joint_positions);
        Eigen::MatrixXd J_pinv = J.completeOrthogonalDecomposition().pseudoInverse();
        Eigen::MatrixXd N = Eigen::MatrixXd::Identity(J.cols(), J.cols()) - J_pinv * J;
        
        // Compute null space gradient for optimization criteria
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(J.cols());
        
        // Add gradient terms for different criteria
        if (joint_limit_avoidance_enabled_)
        {
            gradient += joint_limit_weight_ * computeJointLimitGradient(q);
        }
        
        if (manipulability_weight_ > 0.0)
        {
            gradient += manipulability_weight_ * computeManipulabilityGradient(q);
        }
        
        if (collision_avoidance_weight_ > 0.0 && constraints.collision_avoidance_enabled)
        {
            gradient += collision_avoidance_weight_ *
                       computeCollisionAvoidanceGradient(q, constraints.collision_data);
        }
        
        // Project gradient into null space
        Eigen::VectorXd null_space_gradient = N * gradient;
        
        // Apply null space motion
        double step_size = 0.01;
        for (int i = 0; i < J.cols(); ++i)
        {
            q(i) += step_size * null_space_gradient(i);
        }
        
        solution.redundancy_resolution_applied = true;
        solution.null_space_magnitude = null_space_gradient.norm();
    }
    
    /**
     * @brief Compute joint limit cost
     */
    double computeJointLimitCost(const Eigen::VectorXd& q)
    {
        double cost = 0.0;
        for (int i = 0; i < q.size(); ++i)
        {
            double mid = (joint_position_limits_[i].first + joint_position_limits_[i].second) / 2.0;
            double range = joint_position_limits_[i].second - joint_position_limits_[i].first;
            double normalized_distance = 2.0 * (q(i) - mid) / range;
            
            // Quadratic penalty near limits
            cost += normalized_distance * normalized_distance;
        }
        return cost;
    }
    
    /**
     * @brief Compute collision avoidance cost
     */
    double computeCollisionAvoidanceCost(const Eigen::VectorXd& q,
                                        const CollisionData& collision_data)
    {
        // Simplified implementation - in production, integrate with collision checker
        return 0.0;
    }
    
    /**
     * @brief Compute joint limit gradient
     */
    Eigen::VectorXd computeJointLimitGradient(const KDL::JntArray& q)
    {
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(q.rows());
        
        for (int i = 0; i < q.rows(); ++i)
        {
            double mid = (joint_position_limits_[i].first + joint_position_limits_[i].second) / 2.0;
            double range = joint_position_limits_[i].second - joint_position_limits_[i].first;
            
            // Gradient drives joints toward middle of range
            gradient(i) = (mid - q(i)) / (range * range);
        }
        
        return gradient;
    }
    
    /**
     * @brief Compute manipulability gradient
     */
    Eigen::VectorXd computeManipulabilityGradient(const KDL::JntArray& q)
    {
        // Numerical gradient of manipulability measure
        Eigen::VectorXd gradient = Eigen::VectorXd::Zero(q.rows());
        double epsilon = 1e-6;
        
        for (int i = 0; i < q.rows(); ++i)
        {
            KDL::JntArray q_plus = q;
            KDL::JntArray q_minus = q;
            
            q_plus(i) += epsilon;
            q_minus(i) -= epsilon;
            
            // Convert to vectors for manipulability computation
            std::vector<double> qp_vec(q.rows()), qm_vec(q.rows());
            for (int j = 0; j < q.rows(); ++j)
            {
                qp_vec[j] = q_plus(j);
                qm_vec[j] = q_minus(j);
            }
            
            double m_plus = computeManipulabilityMeasure(qp_vec);
            double m_minus = computeManipulabilityMeasure(qm_vec);
            
            gradient(i) = (m_plus - m_minus) / (2.0 * epsilon);
        }
        
        return gradient;
    }
    
    /**
     * @brief Compute collision avoidance gradient
     */
    Eigen::VectorXd computeCollisionAvoidanceGradient(const KDL::JntArray& q,
                                                     const CollisionData& collision_data)
    {
        // Placeholder - integrate with collision checker in production
        return Eigen::VectorXd::Zero(q.rows());
    }
    
    /**
     * @brief Compute manipulability measure
     */
    void computeManipulability(const KDL::JntArray& q, IKSolution& solution)
    {
        std::vector<double> q_vec(q.rows());
        for (int i = 0; i < q.rows(); ++i)
        {
            q_vec[i] = q(i);
        }
        
        solution.manipulability = computeManipulabilityMeasure(q_vec);
        solution.distance_to_singularity = 1.0 / (solution.manipulability + 1e-6);
    }
    
    /**
     * @brief Check for singularities
     */
    void checkForSingularities(const KDL::JntArray& q, IKSolution& solution)
    {
        std::vector<double> q_vec(q.rows());
        for (int i = 0; i < q.rows(); ++i)
        {
            q_vec[i] = q(i);
        }
        
        SingularityReport report = checkSingularities(q_vec);
        solution.singularity_detected = report.is_singular;
        solution.condition_number = report.condition_number;
        
        if (solution.singularity_detected)
        {
            RCLCPP_WARN(node_->get_logger(),
                       "Singularity detected: condition number = %.2e",
                       report.condition_number);
            solution_stats_.singularity_detections++;
        }
    }
    
    /**
     * @brief Smooth trajectory solutions
     */
    void smoothTrajectorySolutions(std::vector<IKSolution>& solutions)
    {
        if (solutions.size() < 3)
        {
            return;
        }
        
        // Simple moving average smoothing
        for (size_t i = 1; i < solutions.size() - 1; ++i)
        {
            if (solutions[i-1].success && solutions[i].success && solutions[i+1].success)
            {
                size_t dof = solutions[i].joint_positions.size();
                for (size_t j = 0; j < dof; ++j)
                {
                    double smoothed = (solutions[i-1].joint_positions[j] +
                                     solutions[i].joint_positions[j] * 2.0 +
                                     solutions[i+1].joint_positions[j]) / 4.0;
                    solutions[i].joint_positions[j] = smoothed;
                }
            }
        }
    }
    
    /**
     * @brief Adjust Jacobian for reference point
     */
    void adjustJacobianForReferencePoint(const KDL::JntArray& q,
                                        const Eigen::Vector3d& reference_point,
                                        Eigen::MatrixXd& jacobian)
    {
        KDL::Frame tip_frame;
        fk_solver_->JntToCart(q, tip_frame);
        
        KDL::Vector kdl_ref(reference_point.x(), reference_point.y(), reference_point.z());
        KDL::Vector offset = tip_frame.p - kdl_ref;
        
        // Adjust position part of Jacobian for reference point
        for (unsigned int j = 0; j < chain_.getNrOfJoints(); ++j)
        {
            KDL::Vector vel = KDL::Vector(jacobian(0, j), jacobian(1, j), jacobian(2, j));
            KDL::Vector ang = KDL::Vector(jacobian(3, j), jacobian(4, j), jacobian(5, j));
            
            // v_ref = v_tip + ω × (tip - ref)
            KDL::Vector vel_ref = vel + ang * offset;
            
            jacobian(0, j) = vel_ref.x();
            jacobian(1, j) = vel_ref.y();
            jacobian(2, j) = vel_ref.z();
        }
    }
    
    /**
     * @brief Compute frame error
     */
    double computeFrameError(const KDL::Frame& frame1, const KDL::Frame& frame2)
    {
        double position_error = (frame1.p - frame2.p).Norm();
        double orientation_error = computeOrientationError(frame1.M, frame2.M);
        return position_error + orientation_error;
    }
    
    /**
     * @brief Compute orientation error between two rotation matrices
     */
    double computeOrientationError(const KDL::Rotation& R1, const KDL::Rotation& R2)
    {
        KDL::Rotation R_err = R1.Inverse() * R2;
        KDL::Vector axis;
        double angle = R_err.GetRotAngle(axis);
        return std::abs(angle);
    }
    
    /**
     * @brief Convert ROS pose to KDL frame
     */
    KDL::Frame poseToKDLFrame(const geometry_msgs::msg::Pose& pose)
    {
        KDL::Vector position(pose.position.x, pose.position.y, pose.position.z);
        KDL::Rotation rotation = KDL::Rotation::Quaternion(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w);
        return KDL::Frame(rotation, position);
    }
    
    // Member variables
    std::shared_ptr<moveit::core::RobotModel> robot_model_;
    rclcpp::Node::SharedPtr node_;
    std::string group_name_;
    
    // KDL structures
    KDL::Chain chain_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    std::unique_ptr<KDL::ChainIkSolverVel> ik_solver_vel_;
    std::unique_ptr<KDL::ChainIkSolverPos> ik_solver_pos_;
    std::unique_ptr<KDL::ChainJntToJacSolver> jacobian_solver_;
    
    // Solver configuration
    IKSolverType ik_solver_type_;
    int max_iterations_;
    double tolerance_;
    double lambda_;
    double singularity_threshold_;
    
    // Joint limits
    std::vector<std::pair<double, double>> joint_position_limits_;
    std::vector<double> joint_velocity_limits_;
    std::vector<double> joint_acceleration_limits_;
    
    // Redundancy resolution
    bool redundancy_resolution_enabled_;
    bool singularity_avoidance_enabled_;
    bool joint_limit_avoidance_enabled_;
    double collision_avoidance_weight_;
    double manipulability_weight_;
    double joint_limit_weight_;
    double energy_weight_;
    
    // Statistics
    IKSolverStats solution_stats_;
};

// Public interface implementation

InverseKinematics::InverseKinematics(
    const std::shared_ptr<moveit::core::RobotModel>& robot_model,
    const rclcpp::Node::SharedPtr& node,
    const std::string& group_name)
    : impl_(std::make_unique<Impl>(robot_model, node, group_name))
{
}

InverseKinematics::~InverseKinematics() = default;

IKSolution InverseKinematics::solveIK(
    const geometry_msgs::msg::Pose& target_pose,
    const std::vector<double>& initial_guess,
    const IKConstraints& constraints)
{
    return impl_->solveIK(target_pose, initial_guess, constraints);
}

std::vector<IKSolution> InverseKinematics::solveIKTrajectory(
    const std::vector<geometry_msgs::msg::Pose>& target_poses,
    const std::vector<double>& initial_guess,
    const IKConstraints& constraints,
    double continuity_weight)
{
    return impl_->solveIKTrajectory(target_poses, initial_guess, constraints, continuity_weight);
}

Eigen::MatrixXd InverseKinematics::computeJacobian(
    const std::vector<double>& joint_positions,
    const Eigen::Vector3d& reference_point)
{
    return impl_->computeJacobian(joint_positions, reference_point);
}

double InverseKinematics::computeManipulabilityMeasure(
    const std::vector<double>& joint_positions)
{
    return impl_->computeManipulabilityMeasure(joint_positions);
}

SingularityReport InverseKinematics::checkSingularities(
    const std::vector<double>& joint_positions)
{
    return impl_->checkSingularities(joint_positions);
}

void InverseKinematics::setIKSolverType(IKSolverType solver_type)
{
    impl_->setIKSolverType(solver_type);
}

void InverseKinematics::setSolverParameters(
    int max_iterations, double tolerance, double lambda)
{
    impl_->setSolverParameters(max_iterations, tolerance, lambda);
}

void InverseKinematics::enableRedundancyResolution(bool enable)
{
    impl_->enableRedundancyResolution(enable);
}

void InverseKinematics::setRedundancyWeights(const RedundancyWeights& weights)
{
    impl_->setRedundancyWeights(weights);
}

IKSolverStats InverseKinematics::getStatistics() const
{
    return impl_->getStatistics();
}

void InverseKinematics::resetStatistics()
{
    impl_->resetStatistics();
}

} // namespace motion_planning
