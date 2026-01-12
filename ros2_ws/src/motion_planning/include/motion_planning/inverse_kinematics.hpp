// ml-object-picking-robot/ros2_ws/src/motion_planning/include/motion_planning/inverse_kinematics.hpp
// Industrial-grade Inverse Kinematics Solver for 6-DOF Robotic Arms
// KDL-based with multiple solver strategies and singularity handling

#ifndef MOTION_PLANNING__INVERSE_KINEMATICS_HPP_
#define MOTION_PLANNING__INVERSE_KINEMATICS_HPP_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <shared_mutex>
#include <functional>
#include <queue>
#include <limits>

// ROS2
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

// KDL (Kinematics and Dynamics Library)
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/chainiksolverpos_nr.hpp>
#include <kdl/chainiksolverpos_lma.hpp>
#include <kdl/chainiksolvervel_wdls.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolvervel_recursive.hpp>
#include <kdl/frames.hpp>
#include <kdl/jntarray.hpp>
#include <kdl/jacobian.hpp>
#include <kdl_parser/kdl_parser.hpp>

// Eigen for advanced mathematics
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>

// Thread pooling and async operations
#include <thread>
#include <future>
#include <condition_variable>

namespace motion_planning
{

// ============================================
// ENUMS AND CONSTANTS
// ============================================

enum class IKSolverType
{
    KDL_LMA = 0,        // Levenberg-Marquardt (most robust)
    KDL_NR = 1,         // Newton-Raphson (fast)
    KDL_WDLS = 2,       // Weighted Damped Least Squares
    TRAC_IK = 3,        // TRAC-IK (faster alternative)
    IKFAST = 4,         // IKFast (analytic, fastest)
    CUSTOM_GRADIENT = 5,// Custom gradient descent
    HYBRID = 6          // Hybrid solver selection
};

enum class IKSolutionStrategy
{
    SINGLE_SOLUTION = 0,    // Return first valid solution
    MULTIPLE_SOLUTIONS = 1, // Return all valid solutions
    OPTIMAL_SOLUTION = 2,   // Return optimal based on criteria
    CLOSEST_TO_SEED = 3,    // Return closest to seed state
    SAFEST_SOLUTION = 4     // Return safest (maximum margin)
};

enum class IKSolutionStatus
{
    SUCCESS = 0,
    CONVERGENCE_FAILED = 1,
    SINGULARITY_DETECTED = 2,
    JOINT_LIMIT_VIOLATION = 3,
    WORKSPACE_VIOLATION = 4,
    TIMEOUT = 5,
    NUMERICAL_ERROR = 6,
    INVALID_INPUT = 7
};

enum class SingularityType
{
    NONE = 0,
    SHOULDER_SINGULARITY = 1,      // Arm fully extended
    WRIST_SINGULARITY = 2,         // Wrist axes aligned
    ELBOW_SINGULARITY = 3,         // Elbow at limit
    ALGORITHMIC_SINGULARITY = 4,   // Jacobian rank deficient
    BOUNDARY_SINGULARITY = 5       // At workspace boundary
};

struct IKSolverConfig
{
    // Solver selection
    IKSolverType primary_solver;
    IKSolverType fallback_solver;
    IKSolutionStrategy solution_strategy;
    
    // Convergence criteria
    double position_tolerance;      // meters
    double orientation_tolerance;   // radians
    int max_iterations;
    double max_computation_time;    // seconds
    double epsilon;                 // Numerical epsilon
    
    // Damping parameters (for damped solvers)
    double lambda_initial;          // Initial damping factor
    double lambda_factor;           // Damping adjustment factor
    double lambda_max;              // Maximum damping
    double lambda_min;              // Minimum damping
    
    // Weighted parameters (for WDLS)
    Eigen::VectorXd position_weights;
    Eigen::VectorXd orientation_weights;
    Eigen::VectorXd joint_weights;
    
    // Multiple solutions handling
    int max_solutions;
    double solution_tolerance;      // Tolerance between solutions
    bool enable_solution_caching;
    size_t solution_cache_size;
    
    // Singularity handling
    double singularity_threshold;
    double singularity_margin;      // Safety margin from singularities
    bool enable_singularity_avoidance;
    
    // Joint limit handling
    bool enforce_joint_limits;
    bool enforce_velocity_limits;
    bool enforce_torque_limits;
    double joint_limit_margin;      // Safety margin from limits
    
    // Performance optimization
    bool enable_caching;
    size_t cache_size;
    double cache_ttl;               // seconds
    bool enable_parallel_computation;
    int num_threads;
    
    // Monitoring and debugging
    bool enable_statistics;
    bool enable_visualization;
    double statistics_publish_rate;
    
    IKSolverConfig()
        : primary_solver(IKSolverType::KDL_LMA),
          fallback_solver(IKSolverType::KDL_NR),
          solution_strategy(IKSolutionStrategy::OPTIMAL_SOLUTION),
          position_tolerance(0.001),      // 1mm
          orientation_tolerance(0.01),    // ~0.57°
          max_iterations(500),
          max_computation_time(0.1),      // 100ms
          epsilon(1e-10),
          lambda_initial(0.01),
          lambda_factor(10.0),
          lambda_max(1.0),
          lambda_min(1e-12),
          max_solutions(8),
          solution_tolerance(0.01),       // 1cm/0.57°
          enable_solution_caching(true),
          solution_cache_size(10000),
          singularity_threshold(1e-3),
          singularity_margin(0.1),        // 10cm/5.7°
          enable_singularity_avoidance(true),
          enforce_joint_limits(true),
          enforce_velocity_limits(true),
          enforce_torque_limits(false),
          joint_limit_margin(0.0872665),  // 5°
          enable_caching(true),
          cache_size(10000),
          cache_ttl(3600.0),              // 1 hour
          enable_parallel_computation(true),
          num_threads(4),
          enable_statistics(true),
          enable_visualization(false),
          statistics_publish_rate(1.0)
    {
        // Default weights: prioritize position over orientation
        position_weights = Eigen::VectorXd::Ones(3) * 1.0;
        orientation_weights = Eigen::VectorXd::Ones(3) * 0.5;
        joint_weights = Eigen::VectorXd::Ones(6) * 0.1;
    }
};

struct IKSolution
{
    std::vector<double> joint_positions;    // radians
    std::vector<double> joint_velocities;   // rad/s (if computed)
    IKSolutionStatus status;
    double error_position;                  // meters
    double error_orientation;               // radians
    int iterations;
    double computation_time;                // seconds
    double manipulability;                  // Manipulability measure
    double singularity_distance;            // Distance to nearest singularity
    std::vector<double> joint_limit_margins; // Distance to each joint limit
    SingularityType singularity_type;
    
    // Quality metrics
    double energy_metric;                   // Energy consumption estimate
    double smoothness_metric;               // Path smoothness
    double safety_metric;                   // Overall safety score
    
    IKSolution()
        : status(IKSolutionStatus::INVALID_INPUT),
          error_position(std::numeric_limits<double>::max()),
          error_orientation(std::numeric_limits<double>::max()),
          iterations(0),
          computation_time(0.0),
          manipulability(0.0),
          singularity_distance(std::numeric_limits<double>::max()),
          singularity_type(SingularityType::NONE),
          energy_metric(0.0),
          smoothness_metric(0.0),
          safety_metric(0.0) {}
};

struct IKRequest
{
    geometry_msgs::msg::Pose target_pose;
    std::vector<double> seed_state;          // Optional seed
    std::string frame_id;                    // Reference frame
    rclcpp::Time timestamp;
    
    // Constraints
    std::vector<double> joint_weights;       // For weighted solutions
    std::vector<double> joint_biases;        // Preferred joint values
    bool require_precise_position;
    bool require_precise_orientation;
    
    // Solution preferences
    IKSolutionStrategy solution_strategy;
    int max_solutions_requested;
    
    IKRequest()
        : require_precise_position(false),
          require_precise_orientation(false),
          solution_strategy(IKSolutionStrategy::OPTIMAL_SOLUTION),
          max_solutions_requested(1) {}
};

struct IKResponse
{
    std::vector<IKSolution> solutions;
    IKSolutionStatus overall_status;
    std::string error_message;
    rclcpp::Time timestamp;
    double total_computation_time;
    
    // Diagnostics
    size_t cache_hits;
    size_t solver_switches;
    std::vector<std::string> solver_sequence;
    
    IKResponse()
        : overall_status(IKSolutionStatus::INVALID_INPUT),
          total_computation_time(0.0),
          cache_hits(0),
          solver_switches(0) {}
};

struct JointLimits
{
    std::vector<double> lower;      // radians
    std::vector<double> upper;      // radians
    std::vector<double> velocity;   // rad/s
    std::vector<double> acceleration; // rad/s²
    std::vector<double> jerk;       // rad/s³
    std::vector<double> torque;     // Nm
    
    // Soft limits (with safety margins)
    std::vector<double> soft_lower;
    std::vector<double> soft_upper;
    std::vector<double> soft_margin;
    
    JointLimits() {}
    
    explicit JointLimits(size_t num_joints)
    {
        lower.resize(num_joints, -M_PI);
        upper.resize(num_joints, M_PI);
        velocity.resize(num_joints, 2.0);
        acceleration.resize(num_joints, 5.0);
        jerk.resize(num_joints, 50.0);
        torque.resize(num_joints, 100.0);
        soft_lower.resize(num_joints, -M_PI + 0.174533);  // 10° margin
        soft_upper.resize(num_joints, M_PI - 0.174533);
        soft_margin.resize(num_joints, 0.174533);
    }
};

struct SingularityRegion
{
    SingularityType type;
    std::string description;
    std::vector<int> involved_joints;
    Eigen::VectorXd condition_vector;
    double threshold;
    double severity;                // 0-1 scale
    std::function<bool(const std::vector<double>&)> detection_function;
    std::function<void(std::vector<double>&)> avoidance_function;
    
    SingularityRegion()
        : type(SingularityType::NONE),
          threshold(0.1),
          severity(0.0) {}
};

// ============================================
// INVERSE KINEMATICS SOLVER CLASS
// ============================================

class InverseKinematicsSolver : public rclcpp::Node
{
public:
    using SharedPtr = std::shared_ptr<InverseKinematicsSolver>;
    using UniquePtr = std::unique_ptr<InverseKinematicsSolver>;
    
    /**
     * @brief Construct a new Inverse Kinematics Solver object
     * 
     * @param node_name ROS2 node name
     * @param options ROS2 node options
     */
    explicit InverseKinematicsSolver(
        const std::string& node_name = "inverse_kinematics_solver",
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions()
    );
    
    /**
     * @brief Destroy the Inverse Kinematics Solver object
     */
    ~InverseKinematicsSolver();
    
    // ============================================
    // INITIALIZATION AND CONFIGURATION
    // ============================================
    
    /**
     * @brief Initialize the IK solver with configuration
     * 
     * @param config Solver configuration
     * @return true if initialization successful
     */
    bool initialize(const IKSolverConfig& config);
    
    /**
     * @brief Load robot model for IK solving
     * 
     * @param urdf_string URDF description of the robot
     * @param srdf_string SRDF description (optional)
     * @return true if robot model loaded successfully
     */
    bool loadRobotModel(
        const std::string& urdf_string,
        const std::string& srdf_string = ""
    );
    
    /**
     * @brief Set joint limits for the robot
     * 
     * @param limits Joint limits structure
     */
    void setJointLimits(const JointLimits& limits);
    
    /**
     * @brief Update solver configuration at runtime
     * 
     * @param config New configuration
     */
    void updateConfig(const IKSolverConfig& config);
    
    /**
     * @brief Add custom singularity region
     * 
     * @param region Singularity region definition
     */
    void addSingularityRegion(const SingularityRegion& region);
    
    /**
     * @brief Clear all singularity regions
     */
    void clearSingularityRegions();
    
    // ============================================
    // INVERSE KINEMATICS SOLVING
    // ============================================
    
    /**
     * @brief Solve inverse kinematics for target pose
     * 
     * @param request IK request with target pose and constraints
     * @param response IK response with solutions (output)
     * @return true if at least one solution found
     */
    bool solveIK(const IKRequest& request, IKResponse& response);
    
    /**
     * @brief Solve IK for target pose (simplified interface)
     * 
     * @param target_pose Target end-effector pose
     * @param seed_state Initial guess (optional)
     * @param solutions Output vector of solutions
     * @return IKSolutionStatus Overall status
     */
    IKSolutionStatus solveIK(
        const geometry_msgs::msg::Pose& target_pose,
        const std::vector<double>& seed_state,
        std::vector<IKSolution>& solutions
    );
    
    /**
     * @brief Solve IK for multiple target poses (batch processing)
     * 
     * @param target_poses Vector of target poses
     * @param seed_states Vector of seed states
     * @param all_solutions Output vector of solution vectors
     * @return std::vector<IKSolutionStatus> Status for each pose
     */
    std::vector<IKSolutionStatus> solveIKBatch(
        const std::vector<geometry_msgs::msg::Pose>& target_poses,
        const std::vector<std::vector<double>>& seed_states,
        std::vector<std::vector<IKSolution>>& all_solutions
    );
    
    /**
     * @brief Solve IK with velocity constraints (for trajectory generation)
     * 
     * @param target_pose Target pose
     * @param target_twist Target twist (linear/angular velocity)
     * @param current_joints Current joint positions
     * @param solution Output solution with velocities
     * @return IKSolutionStatus Solution status
     */
    IKSolutionStatus solveIKWithVelocity(
        const geometry_msgs::msg::Pose& target_pose,
        const geometry_msgs::msg::Twist& target_twist,
        const std::vector<double>& current_joints,
        IKSolution& solution
    );
    
    /**
     * @brief Solve IK for redundant manipulator (null-space optimization)
     * 
     * @param target_pose Target pose
     * @param seed_state Initial guess
     * @param secondary_task Null-space optimization function
     * @param solution Output solution
     * @return IKSolutionStatus Solution status
     */
    IKSolutionStatus solveRedundantIK(
        const geometry_msgs::msg::Pose& target_pose,
        const std::vector<double>& seed_state,
        const std::function<double(const std::vector<double>&)>& secondary_task,
        IKSolution& solution
    );
    
    // ============================================
    // FORWARD KINEMATICS
    // ============================================
    
    /**
     * @brief Compute forward kinematics for given joint state
     * 
     * @param joint_state Joint positions
     * @param pose Output end-effector pose
     * @return true if computation successful
     */
    bool computeFK(
        const std::vector<double>& joint_state,
        geometry_msgs::msg::Pose& pose
    );
    
    /**
     * @brief Compute forward kinematics for all links
     * 
     * @param joint_state Joint positions
     * @param link_poses Output poses for all links
     * @return true if computation successful
     */
    bool computeFKAllLinks(
        const std::vector<double>& joint_state,
        std::unordered_map<std::string, geometry_msgs::msg::Pose>& link_poses
    );
    
    /**
     * @brief Compute Jacobian matrix for given joint state
     * 
     * @param joint_state Joint positions
     * @param jacobian Output Jacobian matrix (6xN)
     * @return true if computation successful
     */
    bool computeJacobian(
        const std::vector<double>& joint_state,
        Eigen::MatrixXd& jacobian
    );
    
    /**
     * @brief Compute manipulability measure
     * 
     * @param joint_state Joint positions
     * @param manipulability Output manipulability measure
     * @return true if computation successful
     */
    bool computeManipulability(
        const std::vector<double>& joint_state,
        double& manipulability
    );
    
    // ============================================
    // VALIDATION AND SAFETY CHECKS
    // ============================================
    
    /**
     * @brief Validate IK solution
     * 
     * @param solution Solution to validate
     * @param target_pose Target pose for verification
     * @return true if solution is valid
     */
    bool validateSolution(
        const IKSolution& solution,
        const geometry_msgs::msg::Pose& target_pose
    );
    
    /**
     * @brief Check if joint state is within limits
     * 
     * @param joint_state Joint positions
     * @param violated_joints Output indices of violated joints
     * @return true if all joints within limits
     */
    bool checkJointLimits(
        const std::vector<double>& joint_state,
        std::vector<int>& violated_joints
    );
    
    /**
     * @brief Check for singularities at given joint state
     * 
     * @param joint_state Joint positions
     * @param detected_singularities Output detected singularities
     * @return true if no singularities detected
     */
    bool checkSingularities(
        const std::vector<double>& joint_state,
        std::vector<SingularityType>& detected_singularities
    );
    
    /**
     * @brief Compute distance to nearest singularity
     * 
     * @param joint_state Joint positions
     * @param distance Output distance metric
     * @param singularity_type Output type of nearest singularity
     * @return true if computation successful
     */
    bool computeSingularityDistance(
        const std::vector<double>& joint_state,
        double& distance,
        SingularityType& singularity_type
    );
    
    /**
     * @brief Check if target pose is reachable
     * 
     * @param target_pose Target pose
     * @param seed_state Optional seed state
     * @return true if pose is likely reachable
     */
    bool isPoseReachable(
        const geometry_msgs::msg::Pose& target_pose,
        const std::vector<double>& seed_state = {}
    );
    
    // ============================================
    // OPTIMIZATION AND REFINEMENT
    // ============================================
    
    /**
     * @brief Optimize IK solution based on criteria
     * 
     * @param solutions Input solutions
     * @param criteria Optimization criteria string
     * @param optimized_solution Output optimized solution
     * @return true if optimization successful
     */
    bool optimizeSolution(
        const std::vector<IKSolution>& solutions,
        const std::string& criteria,
        IKSolution& optimized_solution
    );
    
    /**
     * @brief Refine solution using gradient descent
     * 
     * @param initial_solution Initial solution
     * @param target_pose Target pose
     * @param refined_solution Output refined solution
     * @return true if refinement improved solution
     */
    bool refineSolution(
        const IKSolution& initial_solution,
        const geometry_msgs::msg::Pose& target_pose,
        IKSolution& refined_solution
    );
    
    /**
     * @brief Smooth transition between joint states
     * 
     * @param start_state Start joint state
     * @param end_state End joint state
     * @param num_points Number of intermediate points
     * @param smoothed_path Output smoothed path
     * @return true if smoothing successful
     */
    bool smoothJointPath(
        const std::vector<double>& start_state,
        const std::vector<double>& end_state,
        size_t num_points,
        std::vector<std::vector<double>>& smoothed_path
    );
    
    // ============================================
    // CACHING AND PERFORMANCE
    // ============================================
    
    /**
     * @brief Clear solution cache
     */
    void clearCache();
    
    /**
     * @brief Get cache statistics
     * 
     * @param hit_rate Output cache hit rate
     * @param size Output current cache size
     * @param effectiveness Output cache effectiveness metric
     */
    void getCacheStatistics(
        double& hit_rate,
        size_t& size,
        double& effectiveness
    ) const;
    
    /**
     * @brief Get solver performance statistics
     * 
     * @param avg_solve_time_ms Average solve time
     * @param success_rate Success rate percentage
     * @param avg_iterations Average iterations per solve
     */
    void getPerformanceStatistics(
        double& avg_solve_time_ms,
        double& success_rate,
        double& avg_iterations
    ) const;
    
    /**
     * @brief Reset performance statistics
     */
    void resetStatistics();
    
    // ============================================
    // VISUALIZATION AND DEBUGGING
    // ============================================
    
    /**
     * @brief Get visualization markers for IK solutions
     * 
     * @param solutions Solutions to visualize
     * @param marker_array Output marker array
     */
    void getVisualizationMarkers(
        const std::vector<IKSolution>& solutions,
        visualization_msgs::msg::MarkerArray& marker_array
    );
    
    /**
     * @brief Get workspace visualization markers
     * 
     * @param resolution Resolution for workspace sampling
     * @param marker_array Output marker array
     */
    void getWorkspaceMarkers(
        double resolution,
        visualization_msgs::msg::MarkerArray& marker_array
    );
    
    /**
     * @brief Get singularity region markers
     * 
     * @param marker_array Output marker array
     */
    void getSingularityMarkers(
        visualization_msgs::msg::MarkerArray& marker_array
    );
    
    // ============================================
    // UTILITY FUNCTIONS
    // ============================================
    
    /**
     * @brief Get number of joints in the robot
     * 
     * @return size_t Number of joints
     */
    size_t getNumJoints() const;
    
    /**
     * @brief Get joint names
     * 
     * @return std::vector<std::string> Joint names
     */
    std::vector<std::string> getJointNames() const;
    
    /**
     * @brief Get robot chain information
     * 
     * @return std::string Chain information string
     */
    std::string getChainInfo() const;
    
    /**
     * @brief Check if solver is initialized
     * 
     * @return true if initialized
     */
    bool isInitialized() const;
    
    /**
     * @brief Enable/disable solver
     * 
     * @param enabled True to enable
     */
    void setEnabled(bool enabled);
    
    /**
     * @brief Check if solver is enabled
     * 
     * @return true if enabled
     */
    bool isEnabled() const;

private:
    // ============================================
    // PRIVATE TYPES
    // ============================================
    
    struct CacheKey
    {
        geometry_msgs::msg::Pose target_pose;
        std::vector<double> seed_state;
        std::string strategy_hash;
        
        bool operator==(const CacheKey& other) const;
    };
    
    struct CacheKeyHash
    {
        size_t operator()(const CacheKey& key) const;
    };
    
    struct CacheEntry
    {
        IKResponse response;
        rclcpp::Time timestamp;
        size_t access_count;
        double average_access_time;
        
        CacheEntry() : access_count(0), average_access_time(0.0) {}
    };
    
    struct SolverPerformance
    {
        IKSolverType solver_type;
        size_t total_calls;
        size_t successful_calls;
        double total_time;
        double avg_iterations;
        std::vector<double> solve_times;
        
        SolverPerformance() 
            : total_calls(0), successful_calls(0), total_time(0.0), avg_iterations(0.0) {}
    };
    
    // ============================================
    // PRIVATE METHODS
    // ============================================
    
    /**
     * @brief Initialize KDL solvers based on configuration
     */
    void initializeSolvers();
    
    /**
     * @brief Load KDL chain from URDF model
     * 
     * @return true if successful
     */
    bool loadKDLChain();
    
    /**
     * @brief Initialize joint limits from URDF
     */
    void initializeJointLimitsFromURDF();
    
    /**
     * @brief Main IK solving method with strategy selection
     * 
     * @param request IK request
     * @param response IK response
     * @return true if solving attempted
     */
    bool solveIKInternal(const IKRequest& request, IKResponse& response);
    
    /**
     * @brief Solve IK using KDL Levenberg-Marquardt solver
     * 
     * @param target_pose Target pose
     * @param seed_state Seed state
     * @param solution Output solution
     * @return IKSolutionStatus Solution status
     */
    IKSolutionStatus solveKDL_LMA(
        const geometry_msgs::msg::Pose& target_pose,
        const std::vector<double>& seed_state,
        IKSolution& solution
    );
    
    /**
     * @brief Solve IK using KDL Newton-Raphson solver
     * 
     * @param target_pose Target pose
     * @param seed_state Seed state
     * @param solution Output solution
     * @return IKSolutionStatus Solution status
     */
    IKSolutionStatus solveKDL_NR(
        const geometry_msgs::msg::Pose& target_pose,
        const std::vector<double>& seed_state,
        IKSolution& solution
    );
    
    /**
     * @brief Solve IK using KDL Weighted Damped Least Squares solver
     * 
     * @param target_pose Target pose
     * @param seed_state Seed state
     * @param solution Output solution
     * @return IKSolutionStatus Solution status
     */
    IKSolutionStatus solveKDL_WDLS(
        const geometry_msgs::msg::Pose& target_pose,
        const std::vector<double>& seed_state,
        IKSolution& solution
    );
    
    /**
     * @brief Solve IK using custom gradient descent
     * 
     * @param target_pose Target pose
     * @param seed_state Seed state
     * @param solution Output solution
     * @return IKSolutionStatus Solution status
     */
    IKSolutionStatus solveCustomGradient(
        const geometry_msgs::msg::Pose& target_pose,
        const std::vector<double>& seed_state,
        IKSolution& solution
    );
    
    /**
     * @brief Solve IK using hybrid approach (multiple solvers)
     * 
     * @param target_pose Target pose
     * @param seed_state Seed state
     * @param solutions Output solutions
     * @return IKSolutionStatus Best status
     */
    IKSolutionStatus solveHybrid(
        const geometry_msgs::msg::Pose& target_pose,
        const std::vector<double>& seed_state,
        std::vector<IKSolution>& solutions
    );
    
    /**
     * @brief Generate multiple seed states for exploration
     * 
     * @param base_seed Base seed state
     * @param num_seeds Number of seeds to generate
     * @return std::vector<std::vector<double>> Generated seeds
     */
    std::vector<std::vector<double>> generateSeedStates(
        const std::vector<double>& base_seed,
        size_t num_seeds
    );
    
    /**
     * @brief Check cache for existing solution
     * 
     * @param key Cache key
     * @param response Output response if found
     * @return true if cache hit
     */
    bool checkCache(const CacheKey& key, IKResponse& response);
    
    /**
     * @brief Update cache with new solution
     * 
     * @param key Cache key
     * @param response Response to cache
     */
    void updateCache(const CacheKey& key, const IKResponse& response);
    
    /**
     * @brief Clean expired cache entries
     */
    void cleanCache();
    
    /**
     * @brief Validate and score IK solution
     * 
     * @param solution Solution to validate
     * @param target_pose Target pose
     * @return double Solution score (higher is better)
     */
    double scoreSolution(
        const IKSolution& solution,
        const geometry_msgs::msg::Pose& target_pose
    );
    
    /**
     * @brief Compute solution quality metrics
     * 
     * @param solution Solution to evaluate
     */
    void computeSolutionMetrics(IKSolution& solution);
    
    /**
     * @brief Detect singularities in joint state
     * 
     * @param joint_state Joint state
     * @param solution Output solution with singularity info
     */
    void detectSingularities(
        const std::vector<double>& joint_state,
        IKSolution& solution
    );
    
    /**
     * @brief Apply joint limit constraints to solution
     * 
     * @param solution Solution to constrain
     * @return true if solution remains valid after constraints
     */
    bool applyJointLimitConstraints(IKSolution& solution);
    
    /**
     * @brief Perform null-space optimization
     * 
     * @param solution Current solution
     * @param secondary_task Secondary task function
     * @param optimized_solution Output optimized solution
     * @return true if optimization improved solution
     */
    bool performNullSpaceOptimization(
        const IKSolution& solution,
        const std::function<double(const std::vector<double>&)>& secondary_task,
        IKSolution& optimized_solution
    );
    
    /**
     * @brief Worker thread for parallel IK solving
     */
    void workerThread();
    
    /**
     * @brief Publish statistics
     */
    void publishStatistics();
    
    /**
     * @brief Publish visualization markers
     */
    void publishVisualization();
    
    // ============================================
    // PRIVATE MEMBER VARIABLES
    // ============================================
    
    // Configuration
    IKSolverConfig config_;
    std::shared_mutex config_mutex_;
    
    // Robot model
    std::shared_ptr<urdf::Model> urdf_model_;
    std::shared_ptr<srdf::Model> srdf_model_;
    KDL::Chain kinematic_chain_;
    std::vector<std::string> joint_names_;
    size_t num_joints_;
    
    // KDL solvers
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_pos_;
    std::unique_ptr<KDL::ChainFkSolverVel_recursive> fk_solver_vel_;
    std::unique_ptr<KDL::ChainIkSolverPos_LMA> ik_solver_lma_;
    std::unique_ptr<KDL::ChainIkSolverPos_NR> ik_solver_nr_;
    std::unique_ptr<KDL::ChainIkSolverVel_wdls> ik_solver_vel_wdls_;
    std::unique_ptr<KDL::ChainJntToJacSolver> jacobian_solver_;
    
    // Joint limits
    JointLimits joint_limits_;
    std::shared_mutex limits_mutex_;
    
    // Singularity regions
    std::vector<SingularityRegion> singularity_regions_;
    std::shared_mutex singularity_mutex_;
    
    // Cache
    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> solution_cache_;
    std::mutex cache_mutex_;
    size_t cache_hits_;
    size_t cache_misses_;
    
    // Thread pool
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::atomic<bool> workers_running_;
    std::atomic<int> active_tasks_;
    
    // Performance monitoring
    mutable std::mutex stats_mutex_;
    std::unordered_map<IKSolverType, SolverPerformance> solver_performance_;
    std::vector<double> solve_times_;
    size_t total_solves_;
    size_t successful_solves_;
    rclcpp::Time last_statistics_publish_;
    
    // State
    std::atomic<bool> enabled_;
    std::atomic<bool> initialized_;
    std::atomic<bool> robot_loaded_;
    
    // ROS2 interfaces
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr performance_pub_;
    
    rclcpp::Service<motion_planning_srvs::srv::SolveIK>::SharedPtr solve_ik_service_;
    rclcpp::Service<motion_planning_srvs::srv::SolveIKBatch>::SharedPtr solve_ik_batch_service_;
    rclcpp::Service<motion_planning_srvs::srv::ComputeFK>::SharedPtr compute_fk_service_;
    rclcpp::Service<motion_planning_srvs::srv::ValidateSolution>::SharedPtr validate_service_;
    
    // Timer for periodic tasks
    rclcpp::TimerBase::SharedPtr statistics_timer_;
    rclcpp::TimerBase::SharedPtr visualization_timer_;
    rclcpp::TimerBase::SharedPtr cache_cleanup_timer_;
    
    // Callback groups for parallel execution
    rclcpp::CallbackGroup::SharedPtr callback_group_;
    
    // ============================================
    // PRIVATE STATIC METHODS
    // ============================================
    
    /**
     * @brief Convert ROS pose to KDL frame
     * 
     * @param pose ROS pose
     * @return KDL::Frame KDL frame
     */
    static KDL::Frame poseToKDLFrame(const geometry_msgs::msg::Pose& pose);
    
    /**
     * @brief Convert KDL frame to ROS pose
     * 
     * @param frame KDL frame
     * @param pose Output ROS pose
     */
    static void kdlFrameToPose(const KDL::Frame& frame, geometry_msgs::msg::Pose& pose);
    
    /**
     * @brief Convert vector to KDL JntArray
     * 
     * @param vec Input vector
     * @param jnt_array Output KDL JntArray
     */
    static void vectorToKDLJntArray(
        const std::vector<double>& vec,
        KDL::JntArray& jnt_array
    );
    
    /**
     * @brief Convert KDL JntArray to vector
     * 
     * @param jnt_array Input KDL JntArray
     * @param vec Output vector
     */
    static void kdlJntArrayToVector(
        const KDL::JntArray& jnt_array,
        std::vector<double>& vec
    );
    
    /**
     * @brief Compute pose error between two poses
     * 
     * @param pose1 First pose
     * @param pose2 Second pose
     * @param position_error Output position error
     * @param orientation_error Output orientation error
     */
    static void computePoseError(
        const geometry_msgs::msg::Pose& pose1,
        const geometry_msgs::msg::Pose& pose2,
        double& position_error,
        double& orientation_error
    );
    
    /**
     * @brief Compute manipulability measure from Jacobian
     * 
     * @param jacobian Jacobian matrix
     * @return double Manipulability measure
     */
    static double computeManipulabilityFromJacobian(const Eigen::MatrixXd& jacobian);
    
    /**
     * @brief Compute condition number of matrix
     * 
     * @param matrix Input matrix
     * @return double Condition number
     */
    static double computeConditionNumber(const Eigen::MatrixXd& matrix);
    
    /**
     * @brief Check for shoulder singularity
     * 
     * @param joint_state Joint state
     * @param threshold Detection threshold
     * @return true if singularity detected
     */
    static bool checkShoulderSingularity(
        const std::vector<double>& joint_state,
        double threshold
    );
    
    /**
     * @brief Check for wrist singularity
     * 
     * @param joint_state Joint state
     * @param threshold Detection threshold
     * @return true if singularity detected
     */
    static bool checkWristSingularity(
        const std::vector<double>& joint_state,
        double threshold
    );
    
    /**
     * @brief Generate random joint state within limits
     * 
     * @param limits Joint limits
     * @return std::vector<double> Random joint state
     */
    static std::vector<double> generateRandomJointState(const JointLimits& limits);
    
    /**
     * @brief Interpolate between two joint states
     * 
     * @param start Start state
     * @param end End state
     * @param t Interpolation parameter [0, 1]
     * @return std::vector<double> Interpolated state
     */
    static std::vector<double> interpolateJointStates(
        const std::vector<double>& start,
        const std::vector<double>& end,
        double t
    );
};

}  // namespace motion_planning

// Specialize std::hash for CacheKey
namespace std
{
    template<>
    struct hash<motion_planning::InverseKinematicsSolver::CacheKey>
    {
        size_t operator()(const motion_planning::InverseKinematicsSolver::CacheKey& key) const
        {
            motion_planning::InverseKinematicsSolver::CacheKeyHash hasher;
            return hasher(key);
        }
    };
}

#endif  // MOTION_PLANNING__INVERSE_KINEMATICS_HPP_
