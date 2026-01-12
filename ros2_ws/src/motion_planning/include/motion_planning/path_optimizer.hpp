// ml-object-picking-robot/ros2_ws/src/motion_planning/include/motion_planning/path_optimizer.hpp
// Industrial-grade Path Optimization for Robotic Arm Trajectories
// Time-optimal, energy-efficient, and collision-free path optimization

#ifndef MOTION_PLANNING__PATH_OPTIMIZER_HPP_
#define MOTION_PLANNING__PATH_OPTIMIZER_HPP_

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
#include <deque>

// ROS2
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/path.hpp>

// Eigen for advanced mathematics
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/StdVector>

// OMPL for motion planning
#include <ompl/base/StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/geometric/PathSimplifier.h>

// Optimization libraries
#include <ifopt/problem.h>
#include <ifopt/ipopt_solver.h>
#include <ifopt/variable_set.h>
#include <ifopt/constraint_set.h>
#include <ifopt/cost_term.h>

// Thread pooling
#include <thread>
#include <future>
#include <condition_variable>

namespace motion_planning
{

// ============================================
// ENUMS AND CONSTANTS
// ============================================

enum class OptimizationObjective
{
    TIME_OPTIMAL = 0,           // Minimize execution time
    ENERGY_EFFICIENT = 1,       // Minimize energy consumption
    PATH_LENGTH = 2,            // Minimize path length
    SMOOTHNESS = 3,             // Maximize smoothness (minimize jerk)
    COMBINED = 4,               // Weighted combination
    SAFETY_OPTIMIZED = 5        // Maximize safety margins
};

enum class OptimizationMethod
{
    SHORTCUTTING = 0,           // Path shortcutting
    SIMPLIFICATION = 1,         // Path simplification
    SMOOTHING = 2,              // Path smoothing
    TIME_OPTIMAL_PARAM = 3,     // Time-optimal parameterization
    NONLINEAR_OPTIMIZATION = 4, // Nonlinear optimization
    GRADIENT_DESCENT = 5,       // Gradient descent
    GENETIC_ALGORITHM = 6,      // Genetic algorithm (experimental)
    HYBRID = 7                  // Hybrid approach
};

enum class OptimizationStatus
{
    SUCCESS = 0,
    CONVERGENCE_FAILED = 1,
    CONSTRAINT_VIOLATION = 2,
    TIMEOUT = 3,
    NUMERICAL_ERROR = 4,
    COLLISION_DETECTED = 5,
    INVALID_INPUT = 6,
    LOCAL_MINIMA = 7
};

enum class SmoothingMethod
{
    CHOP = 0,                   // Constrained Hamiltonian Optimization for Motion Planning
    B_SPLINE = 1,               // B-spline interpolation
    CUBIC_SPLINE = 2,           // Cubic spline interpolation
    BEZIER = 3,                 // Bezier curve smoothing
    GAUSSIAN = 4,               // Gaussian smoothing
    SAVITZKY_GOLAY = 5          // Savitzky-Golay filter
};

struct OptimizationConfig
{
    // General optimization settings
    OptimizationObjective objective;
    OptimizationMethod primary_method;
    std::vector<OptimizationMethod> fallback_methods;
    
    // Weights for combined objectives
    double weight_time;
    double weight_energy;
    double weight_path_length;
    double weight_smoothness;
    double weight_safety;
    
    // Convergence criteria
    double tolerance;
    int max_iterations;
    double max_computation_time;    // seconds
    double improvement_threshold;   // Minimum improvement to continue
    
    // Shortcutting parameters
    double max_shortcut_length;     // meters in joint space
    int shortcut_max_iterations;
    double shortcut_success_rate;   // Required success rate
    
    // Simplification parameters
    double simplification_tolerance; // meters
    int simplification_max_points;
    bool simplify_preserve_endpoints;
    
    // Smoothing parameters
    SmoothingMethod smoothing_method;
    double smoothing_factor;
    int smoothing_iterations;
    double max_smoothing_deviation; // meters
    
    // Time-optimal parameterization
    bool enable_time_optimization;
    double max_velocity_scale;
    double max_acceleration_scale;
    double max_jerk_scale;
    bool enforce_torque_limits;
    
    // Safety constraints
    double min_clearance;           // meters
    double safety_margin;
    bool enforce_collision_constraints;
    bool enforce_joint_limit_constraints;
    bool enforce_velocity_constraints;
    
    // Nonlinear optimization (IPOPT)
    double ipopt_tolerance;
    int ipopt_max_iterations;
    std::string ipopt_linear_solver;
    bool ipopt_hessian_approximation;
    
    // Performance optimization
    bool enable_caching;
    size_t cache_size;
    double cache_ttl;               // seconds
    bool enable_parallel_optimization;
    int num_threads;
    
    // Monitoring and debugging
    bool enable_statistics;
    bool enable_visualization;
    double statistics_publish_rate;
    
    OptimizationConfig()
        : objective(OptimizationObjective::COMBINED),
          primary_method(OptimizationMethod::HYBRID),
          weight_time(0.4),
          weight_energy(0.2),
          weight_path_length(0.2),
          weight_smoothness(0.1),
          weight_safety(0.1),
          tolerance(1e-4),
          max_iterations(100),
          max_computation_time(2.0),
          improvement_threshold(0.01),      // 1% improvement
          max_shortcut_length(0.5),
          shortcut_max_iterations(100),
          shortcut_success_rate(0.8),
          simplification_tolerance(0.01),   // 1cm
          simplification_max_points(100),
          simplify_preserve_endpoints(true),
          smoothing_method(SmoothingMethod::B_SPLINE),
          smoothing_factor(0.1),
          smoothing_iterations(50),
          max_smoothing_deviation(0.02),    // 2cm
          enable_time_optimization(true),
          max_velocity_scale(0.8),          // 80% of limits
          max_acceleration_scale(0.7),      // 70% of limits
          max_jerk_scale(0.6),              // 60% of limits
          enforce_torque_limits(false),
          min_clearance(0.05),              // 5cm
          safety_margin(0.02),              // 2cm
          enforce_collision_constraints(true),
          enforce_joint_limit_constraints(true),
          enforce_velocity_constraints(true),
          ipopt_tolerance(1e-6),
          ipopt_max_iterations(200),
          ipopt_linear_solver("mumps"),
          ipopt_hessian_approximation(true),
          enable_caching(true),
          cache_size(5000),
          cache_ttl(1800.0),                // 30 minutes
          enable_parallel_optimization(true),
          num_threads(4),
          enable_statistics(true),
          enable_visualization(false),
          statistics_publish_rate(1.0) {}
};

struct PathOptimizationRequest
{
    trajectory_msgs::msg::JointTrajectory input_trajectory;
    std::vector<double> velocity_limits;    // rad/s
    std::vector<double> acceleration_limits;// rad/s²
    std::vector<double> jerk_limits;        // rad/s³
    std::vector<double> torque_limits;      // Nm
    
    // Constraints
    bool enforce_collision_avoidance;
    std::vector<geometry_msgs::msg::Pose> collision_objects;
    double required_clearance;
    
    // Optimization preferences
    OptimizationObjective objective;
    std::vector<OptimizationMethod> methods;
    double time_weight;
    double energy_weight;
    
    // Validation
    bool validate_after_optimization;
    double validation_tolerance;
    
    PathOptimizationRequest()
        : enforce_collision_avoidance(true),
          required_clearance(0.05),
          objective(OptimizationObjective::COMBINED),
          time_weight(0.4),
          energy_weight(0.2),
          validate_after_optimization(true),
          validation_tolerance(0.001) {}
};

struct PathOptimizationResult
{
    trajectory_msgs::msg::JointTrajectory optimized_trajectory;
    OptimizationStatus status;
    std::string message;
    
    // Performance metrics
    double original_duration;
    double optimized_duration;
    double original_energy;
    double optimized_energy;
    double original_path_length;
    double optimized_path_length;
    double smoothness_improvement;
    double clearance_improvement;
    
    // Computation statistics
    double computation_time;                // seconds
    int iterations;
    size_t waypoints_removed;
    size_t waypoints_added;
    
    // Validation results
    bool collision_free;
    bool within_limits;
    double max_clearance_violation;
    double max_limit_violation;
    
    PathOptimizationResult()
        : status(OptimizationStatus::INVALID_INPUT),
          original_duration(0.0),
          optimized_duration(0.0),
          original_energy(0.0),
          optimized_energy(0.0),
          original_path_length(0.0),
          optimized_path_length(0.0),
          smoothness_improvement(0.0),
          clearance_improvement(0.0),
          computation_time(0.0),
          iterations(0),
          waypoints_removed(0),
          waypoints_added(0),
          collision_free(false),
          within_limits(false),
          max_clearance_violation(0.0),
          max_limit_violation(0.0) {}
};

struct PathQualityMetrics
{
    // Time-based metrics
    double total_duration;
    double average_velocity;
    double max_velocity;
    double velocity_consistency;
    
    // Energy metrics
    double total_energy;
    double average_power;
    double peak_power;
    double energy_efficiency;
    
    // Smoothness metrics
    double total_jerk;
    double max_jerk;
    double jerk_consistency;
    double curvature_variation;
    
    // Safety metrics
    double min_clearance;
    double average_clearance;
    double clearance_consistency;
    double risk_score;
    
    // Geometric metrics
    double path_length;
    double directness;                      // Ratio of straight-line to actual path
    double tortuosity;                      // Complexity of path
    double waypoint_efficiency;
    
    PathQualityMetrics()
        : total_duration(0.0),
          average_velocity(0.0),
          max_velocity(0.0),
          velocity_consistency(0.0),
          total_energy(0.0),
          average_power(0.0),
          peak_power(0.0),
          energy_efficiency(0.0),
          total_jerk(0.0),
          max_jerk(0.0),
          jerk_consistency(0.0),
          curvature_variation(0.0),
          min_clearance(0.0),
          average_clearance(0.0),
          clearance_consistency(0.0),
          risk_score(0.0),
          path_length(0.0),
          directness(0.0),
          tortuosity(0.0),
          waypoint_efficiency(0.0) {}
};

struct OptimizationConstraint
{
    std::string name;
    std::function<double(const std::vector<double>&)> constraint_function;
    double lower_bound;
    double upper_bound;
    bool is_equality;
    double weight;
    
    OptimizationConstraint()
        : lower_bound(-std::numeric_limits<double>::infinity()),
          upper_bound(std::numeric_limits<double>::infinity()),
          is_equality(false),
          weight(1.0) {}
};

// ============================================
// PATH OPTIMIZER CLASS
// ============================================

class PathOptimizer : public rclcpp::Node
{
public:
    using SharedPtr = std::shared_ptr<PathOptimizer>;
    using UniquePtr = std::unique_ptr<PathOptimizer>;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    /**
     * @brief Construct a new Path Optimizer object
     * 
     * @param node_name ROS2 node name
     * @param options ROS2 node options
     */
    explicit PathOptimizer(
        const std::string& node_name = "path_optimizer",
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions()
    );
    
    /**
     * @brief Destroy the Path Optimizer object
     */
    ~PathOptimizer();
    
    // ============================================
    // INITIALIZATION AND CONFIGURATION
    // ============================================
    
    /**
     * @brief Initialize the path optimizer with configuration
     * 
     * @param config Optimization configuration
     * @return true if initialization successful
     */
    bool initialize(const OptimizationConfig& config);
    
    /**
     * @brief Set collision checker for optimization
     * 
     * @param collision_checker Shared pointer to collision checker
     */
    void setCollisionChecker(std::shared_ptr<class CollisionChecker> collision_checker);
    
    /**
     * @brief Set joint limits for optimization constraints
     * 
     * @param lower_limits Lower joint limits (rad)
     * @param upper_limits Upper joint limits (rad)
     * @param velocity_limits Velocity limits (rad/s)
     * @param acceleration_limits Acceleration limits (rad/s²)
     */
    void setJointLimits(
        const std::vector<double>& lower_limits,
        const std::vector<double>& upper_limits,
        const std::vector<double>& velocity_limits,
        const std::vector<double>& acceleration_limits
    );
    
    /**
     * @brief Add custom optimization constraint
     * 
     * @param constraint Constraint definition
     */
    void addConstraint(const OptimizationConstraint& constraint);
    
    /**
     * @brief Clear all custom constraints
     */
    void clearConstraints();
    
    /**
     * @brief Update optimization configuration at runtime
     * 
     * @param config New configuration
     */
    void updateConfig(const OptimizationConfig& config);
    
    // ============================================
    // PATH OPTIMIZATION METHODS
    // ============================================
    
    /**
     * @brief Optimize trajectory using specified methods
     * 
     * @param request Optimization request
     * @param result Optimization result (output)
     * @return true if optimization attempted
     */
    bool optimizePath(const PathOptimizationRequest& request, PathOptimizationResult& result);
    
    /**
     * @brief Optimize trajectory (simplified interface)
     * 
     * @param trajectory Input trajectory
     * @param optimized_trajectory Output optimized trajectory
     * @param method Optimization method to use
     * @return OptimizationStatus Optimization status
     */
    OptimizationStatus optimizePath(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        trajectory_msgs::msg::JointTrajectory& optimized_trajectory,
        OptimizationMethod method = OptimizationMethod::HYBRID
    );
    
    /**
     * @brief Optimize multiple trajectories (batch processing)
     * 
     * @param trajectories Input trajectories
     * @param optimized_trajectories Output optimized trajectories
     * @param methods Optimization methods for each trajectory
     * @return std::vector<OptimizationStatus> Status for each trajectory
     */
    std::vector<OptimizationStatus> optimizePathsBatch(
        const std::vector<trajectory_msgs::msg::JointTrajectory>& trajectories,
        std::vector<trajectory_msgs::msg::JointTrajectory>& optimized_trajectories,
        const std::vector<OptimizationMethod>& methods
    );
    
    /**
     * @brief Apply path shortcutting optimization
     * 
     * @param trajectory Input trajectory
     * @param optimized_trajectory Output trajectory
     * @param max_shortcut_length Maximum shortcut length
     * @param max_iterations Maximum iterations
     * @return OptimizationStatus Optimization status
     */
    OptimizationStatus shortcutPath(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        trajectory_msgs::msg::JointTrajectory& optimized_trajectory,
        double max_shortcut_length,
        int max_iterations
    );
    
    /**
     * @brief Apply path simplification
     * 
     * @param trajectory Input trajectory
     * @param optimized_trajectory Output trajectory
     * @param tolerance Simplification tolerance
     * @param preserve_endpoints Preserve start and end points
     * @return OptimizationStatus Optimization status
     */
    OptimizationStatus simplifyPath(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        trajectory_msgs::msg::JointTrajectory& optimized_trajectory,
        double tolerance,
        bool preserve_endpoints
    );
    
    /**
     * @brief Apply path smoothing
     * 
     * @param trajectory Input trajectory
     * @param optimized_trajectory Output trajectory
     * @param method Smoothing method
     * @param smoothing_factor Smoothing factor
     * @param max_iterations Maximum smoothing iterations
     * @return OptimizationStatus Optimization status
     */
    OptimizationStatus smoothPath(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        trajectory_msgs::msg::JointTrajectory& optimized_trajectory,
        SmoothingMethod method,
        double smoothing_factor,
        int max_iterations
    );
    
    /**
     * @brief Apply time-optimal parameterization
     * 
     * @param trajectory Input trajectory
     * @param optimized_trajectory Output trajectory
     * @param velocity_limits Joint velocity limits
     * @param acceleration_limits Joint acceleration limits
     * @param jerk_limits Joint jerk limits
     * @return OptimizationStatus Optimization status
     */
    OptimizationStatus timeOptimalParameterization(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        trajectory_msgs::msg::JointTrajectory& optimized_trajectory,
        const std::vector<double>& velocity_limits,
        const std::vector<double>& acceleration_limits,
        const std::vector<double>& jerk_limits
    );
    
    /**
     * @brief Apply nonlinear optimization (IPOPT)
     * 
     * @param trajectory Input trajectory
     * @param optimized_trajectory Output trajectory
     * @param objective Optimization objective
     * @param constraints Additional constraints
     * @return OptimizationStatus Optimization status
     */
    OptimizationStatus nonlinearOptimization(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        trajectory_msgs::msg::JointTrajectory& optimized_trajectory,
        OptimizationObjective objective,
        const std::vector<OptimizationConstraint>& constraints
    );
    
    // ============================================
    // PATH ANALYSIS AND METRICS
    // ============================================
    
    /**
     * @brief Compute path quality metrics
     * 
     * @param trajectory Input trajectory
     * @param metrics Output quality metrics
     * @return true if computation successful
     */
    bool computePathMetrics(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        PathQualityMetrics& metrics
    );
    
    /**
     * @brief Compare two trajectories
     * 
     * @param trajectory1 First trajectory
     * @param trajectory2 Second trajectory
     * @param improvement_metrics Output improvement metrics
     * @return true if comparison successful
     */
    bool compareTrajectories(
        const trajectory_msgs::msg::JointTrajectory& trajectory1,
        const trajectory_msgs::msg::JointTrajectory& trajectory2,
        std::unordered_map<std::string, double>& improvement_metrics
    );
    
    /**
     * @brief Validate trajectory for constraints
     * 
     * @param trajectory Trajectory to validate
     * @param check_collisions Check for collisions
     * @param check_limits Check joint limits
     * @param check_continuity Check continuity
     * @return true if trajectory is valid
     */
    bool validateTrajectory(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        bool check_collisions = true,
        bool check_limits = true,
        bool check_continuity = true
    );
    
    /**
     * @brief Compute clearance along trajectory
     * 
     * @param trajectory Input trajectory
     * @param clearances Output clearance values at each point
     * @param min_clearance Output minimum clearance
     * @param avg_clearance Output average clearance
     * @return true if computation successful
     */
    bool computeTrajectoryClearance(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        std::vector<double>& clearances,
        double& min_clearance,
        double& avg_clearance
    );
    
    /**
     * @brief Compute energy consumption of trajectory
     * 
     * @param trajectory Input trajectory
     * @param total_energy Output total energy consumption
     * @param peak_power Output peak power
     * @return true if computation successful
     */
    bool computeTrajectoryEnergy(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        double& total_energy,
        double& peak_power
    );
    
    // ============================================
    // UTILITY FUNCTIONS
    // ============================================
    
    /**
     * @brief Resample trajectory to uniform time intervals
     * 
     * @param trajectory Input trajectory
     * @param dt Desired time step
     * @param resampled_trajectory Output resampled trajectory
     * @return true if resampling successful
     */
    bool resampleTrajectory(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        double dt,
        trajectory_msgs::msg::JointTrajectory& resampled_trajectory
    );
    
    /**
     * @brief Convert joint trajectory to Cartesian path
     * 
     * @param trajectory Joint trajectory
     * @param cartesian_path Output Cartesian path
     * @return true if conversion successful
     */
    bool jointToCartesianPath(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        nav_msgs::msg::Path& cartesian_path
    );
    
    /**
     * @brief Blend two trajectories smoothly
     * 
     * @param trajectory1 First trajectory
     * @param trajectory2 Second trajectory
     * @param blend_radius Blending radius
     * @param blended_trajectory Output blended trajectory
     * @return true if blending successful
     */
    bool blendTrajectories(
        const trajectory_msgs::msg::JointTrajectory& trajectory1,
        const trajectory_msgs::msg::JointTrajectory& trajectory2,
        double blend_radius,
        trajectory_msgs::msg::JointTrajectory& blended_trajectory
    );
    
    /**
     * @brief Generate minimum jerk trajectory between points
     * 
     * @param start_point Start joint state
     * @param end_point End joint state
     * @param duration Trajectory duration
     * @param num_points Number of points
     * @param trajectory Output trajectory
     * @return true if generation successful
     */
    bool generateMinimumJerkTrajectory(
        const trajectory_msgs::msg::JointTrajectoryPoint& start_point,
        const trajectory_msgs::msg::JointTrajectoryPoint& end_point,
        double duration,
        size_t num_points,
        trajectory_msgs::msg::JointTrajectory& trajectory
    );
    
    // ============================================
    // CACHING AND PERFORMANCE
    // ============================================
    
    /**
     * @brief Clear optimization cache
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
     * @brief Get optimization performance statistics
     * 
     * @param avg_optimization_time_ms Average optimization time
     * @param success_rate Success rate percentage
     * @param avg_improvement Average improvement percentage
     */
    void getPerformanceStatistics(
        double& avg_optimization_time_ms,
        double& success_rate,
        double& avg_improvement
    ) const;
    
    /**
     * @brief Reset performance statistics
     */
    void resetStatistics();
    
    // ============================================
    // VISUALIZATION AND DEBUGGING
    // ============================================
    
    /**
     * @brief Get visualization markers for trajectory
     * 
     * @param trajectory Trajectory to visualize
     * @param marker_array Output marker array
     * @param color Color for visualization
     */
    void getTrajectoryMarkers(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        visualization_msgs::msg::MarkerArray& marker_array,
        const std::array<double, 4>& color = {1.0, 0.0, 0.0, 1.0}
    );
    
    /**
     * @brief Get clearance visualization markers
     * 
     * @param trajectory Trajectory to visualize
     * @param marker_array Output marker array
     */
    void getClearanceMarkers(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        visualization_msgs::msg::MarkerArray& marker_array
    );
    
    /**
     * @brief Get optimization progress markers
     * 
     * @param iteration Iteration number
     * @param progress Optimization progress data
     * @param marker_array Output marker array
     */
    void getOptimizationProgressMarkers(
        int iteration,
        const std::vector<double>& progress,
        visualization_msgs::msg::MarkerArray& marker_array
    );
    
    // ============================================
    // CONFIGURATION AND STATE
    // ============================================
    
    /**
     * @brief Get current optimization configuration
     * 
     * @return OptimizationConfig Current configuration
     */
    OptimizationConfig getConfig() const;
    
    /**
     * @brief Check if optimizer is initialized
     * 
     * @return true if initialized
     */
    bool isInitialized() const;
    
    /**
     * @brief Enable/disable optimizer
     * 
     * @param enabled True to enable
     */
    void setEnabled(bool enabled);
    
    /**
     * @brief Check if optimizer is enabled
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
        trajectory_msgs::msg::JointTrajectory trajectory;
        OptimizationObjective objective;
        OptimizationMethod method;
        std::string constraints_hash;
        
        bool operator==(const CacheKey& other) const;
    };
    
    struct CacheKeyHash
    {
        size_t operator()(const CacheKey& key) const;
    };
    
    struct CacheEntry
    {
        PathOptimizationResult result;
        rclcpp::Time timestamp;
        size_t access_count;
        double average_access_time;
        
        CacheEntry() : access_count(0), average_access_time(0.0) {}
    };
    
    struct OptimizationTask
    {
        PathOptimizationRequest request;
        std::promise<PathOptimizationResult> promise;
        rclcpp::Time submission_time;
        
        OptimizationTask(PathOptimizationRequest req, std::promise<PathOptimizationResult> prom)
            : request(std::move(req)), promise(std::move(prom)) {}
    };
    
    struct TrajectoryWaypoint
    {
        std::vector<double> positions;
        std::vector<double> velocities;
        std::vector<double> accelerations;
        double time_from_start;
        
        TrajectoryWaypoint() : time_from_start(0.0) {}
        
        explicit TrajectoryWaypoint(size_t num_joints)
            : positions(num_joints, 0.0),
              velocities(num_joints, 0.0),
              accelerations(num_joints, 0.0),
              time_from_start(0.0) {}
    };
    
    // ============================================
    // PRIVATE METHODS
    // ============================================
    
    /**
     * @brief Initialize optimization components
     */
    void initializeComponents();
    
    /**
     * @brief Main optimization method with strategy selection
     * 
     * @param request Optimization request
     * @param result Optimization result
     * @return true if optimization attempted
     */
    bool optimizePathInternal(const PathOptimizationRequest& request, PathOptimizationResult& result);
    
    /**
     * @brief Apply hybrid optimization (multiple methods)
     * 
     * @param trajectory Input trajectory
     * @param optimized_trajectory Output trajectory
     * @param config Optimization configuration
     * @param result Optimization result
     * @return OptimizationStatus Optimization status
     */
    OptimizationStatus applyHybridOptimization(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        trajectory_msgs::msg::JointTrajectory& optimized_trajectory,
        const OptimizationConfig& config,
        PathOptimizationResult& result
    );
    
    /**
     * @brief Apply shortcutting algorithm
     * 
     * @param waypoints Input waypoints
     * @param optimized_waypoints Output waypoints
     * @param config Optimization configuration
     * @return true if optimization successful
     */
    bool applyShortcutting(
        const std::vector<TrajectoryWaypoint>& waypoints,
        std::vector<TrajectoryWaypoint>& optimized_waypoints,
        const OptimizationConfig& config
    );
    
    /**
     * @brief Apply simplification algorithm
     * 
     * @param waypoints Input waypoints
     * @param optimized_waypoints Output waypoints
     * @param config Optimization configuration
     * @return true if optimization successful
     */
    bool applySimplification(
        const std::vector<TrajectoryWaypoint>& waypoints,
        std::vector<TrajectoryWaypoint>& optimized_waypoints,
        const OptimizationConfig& config
    );
    
    /**
     * @brief Apply smoothing algorithm
     * 
     * @param waypoints Input waypoints
     * @param optimized_waypoints Output waypoints
     * @param config Optimization configuration
     * @return true if optimization successful
     */
    bool applySmoothing(
        const std::vector<TrajectoryWaypoint>& waypoints,
        std::vector<TrajectoryWaypoint>& optimized_waypoints,
        const OptimizationConfig& config
    );
    
    /**
     * @brief Apply B-spline smoothing
     * 
     * @param waypoints Input waypoints
     * @param optimized_waypoints Output waypoints
     * @param smoothing_factor Smoothing factor
     * @param iterations Number of iterations
     * @return true if smoothing successful
     */
    bool applyBSplineSmoothing(
        const std::vector<TrajectoryWaypoint>& waypoints,
        std::vector<TrajectoryWaypoint>& optimized_waypoints,
        double smoothing_factor,
        int iterations
    );
    
    /**
     * @brief Apply cubic spline smoothing
     * 
     * @param waypoints Input waypoints
     * @param optimized_waypoints Output waypoints
     * @return true if smoothing successful
     */
    bool applyCubicSplineSmoothing(
        const std::vector<TrajectoryWaypoint>& waypoints,
        std::vector<TrajectoryWaypoint>& optimized_waypoints
    );
    
    /**
     * @brief Apply CHOMP smoothing
     * 
     * @param waypoints Input waypoints
     * @param optimized_waypoints Output waypoints
     * @param smoothing_factor Smoothing factor
     * @param iterations Number of iterations
     * @return true if smoothing successful
     */
    bool applyChompSmoothing(
        const std::vector<TrajectoryWaypoint>& waypoints,
        std::vector<TrajectoryWaypoint>& optimized_waypoints,
        double smoothing_factor,
        int iterations
    );
    
    /**
     * @brief Apply time-optimal parameterization
     * 
     * @param waypoints Input waypoints (positions only)
     * @param optimized_waypoints Output waypoints (with timing)
     * @param velocity_limits Velocity limits
     * @param acceleration_limits Acceleration limits
     * @param jerk_limits Jerk limits
     * @return true if parameterization successful
     */
    bool applyTimeOptimalParameterization(
        const std::vector<TrajectoryWaypoint>& waypoints,
        std::vector<TrajectoryWaypoint>& optimized_waypoints,
        const std::vector<double>& velocity_limits,
        const std::vector<double>& acceleration_limits,
        const std::vector<double>& jerk_limits
    );
    
    /**
     * @brief Apply nonlinear optimization using IPOPT
     * 
     * @param waypoints Input waypoints
     * @param optimized_waypoints Output waypoints
     * @param config Optimization configuration
     * @param constraints Additional constraints
     * @return OptimizationStatus Optimization status
     */
    OptimizationStatus applyNonlinearOptimization(
        const std::vector<TrajectoryWaypoint>& waypoints,
        std::vector<TrajectoryWaypoint>& optimized_waypoints,
        const OptimizationConfig& config,
        const std::vector<OptimizationConstraint>& constraints
    );
    
    /**
     * @brief Check cache for existing optimization result
     * 
     * @param key Cache key
     * @param result Output result if found
     * @return true if cache hit
     */
    bool checkCache(const CacheKey& key, PathOptimizationResult& result);
    
    /**
     * @brief Update cache with new optimization result
     * 
     * @param key Cache key
     * @param result Result to cache
     */
    void updateCache(const CacheKey& key, const PathOptimizationResult& result);
    
    /**
     * @brief Clean expired cache entries
     */
    void cleanCache();
    
    /**
     * @brief Convert trajectory to waypoints
     * 
     * @param trajectory Input trajectory
     * @param waypoints Output waypoints
     * @return true if conversion successful
     */
    bool trajectoryToWaypoints(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        std::vector<TrajectoryWaypoint>& waypoints
    );
    
    /**
     * @brief Convert waypoints to trajectory
     * 
     * @param waypoints Input waypoints
     * @param trajectory Output trajectory
     * @return true if conversion successful
     */
    bool waypointsToTrajectory(
        const std::vector<TrajectoryWaypoint>& waypoints,
        trajectory_msgs::msg::JointTrajectory& trajectory
    );
    
    /**
     * @brief Compute trajectory cost based on objective
     * 
     * @param waypoints Trajectory waypoints
     * @param objective Optimization objective
     * @param cost Output cost value
     * @return true if computation successful
     */
    bool computeTrajectoryCost(
        const std::vector<TrajectoryWaypoint>& waypoints,
        OptimizationObjective objective,
        double& cost
    );
    
    /**
     * @brief Check waypoint for collisions
     * 
     * @param waypoint Waypoint to check
     * @param clearance Output clearance value
     * @return true if collision-free
     */
    bool checkWaypointCollision(
        const TrajectoryWaypoint& waypoint,
        double& clearance
    );
    
    /**
     * @brief Check waypoint for joint limit violations
     * 
     * @param waypoint Waypoint to check
     * @param violation Output violation amount
     * @return true if within limits
     */
    bool checkWaypointLimits(
        const TrajectoryWaypoint& waypoint,
        double& violation
    );
    
    /**
     * @brief Compute smoothness metric for trajectory
     * 
     * @param waypoints Trajectory waypoints
     * @param smoothness Output smoothness metric
     * @return true if computation successful
     */
    bool computeSmoothnessMetric(
        const std::vector<TrajectoryWaypoint>& waypoints,
        double& smoothness
    );
    
    /**
     * @brief Worker thread for parallel optimization
     */
    void workerThread();
    
    /**
     * @brief Process optimization task
     * 
     * @param task Optimization task
     */
    void processOptimizationTask(std::shared_ptr<OptimizationTask> task);
    
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
    OptimizationConfig config_;
    std::shared_mutex config_mutex_;
    
    // Dependencies
    std::shared_ptr<class CollisionChecker> collision_checker_;
    std::shared_ptr<class InverseKinematicsSolver> ik_solver_;
    
    // Joint limits and constraints
    std::vector<double> joint_lower_limits_;
    std::vector<double> joint_upper_limits_;
    std::vector<double> joint_velocity_limits_;
    std::vector<double> joint_acceleration_limits_;
    std::vector<double> joint_jerk_limits_;
    std::vector<double> joint_torque_limits_;
    
    std::vector<OptimizationConstraint> custom_constraints_;
    std::shared_mutex constraints_mutex_;
    
    // Cache
    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> optimization_cache_;
    std::mutex cache_mutex_;
    size_t cache_hits_;
    size_t cache_misses_;
    
    // Thread pool
    std::vector<std::thread> worker_threads_;
    std::queue<std::shared_ptr<OptimizationTask>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_condition_;
    std::atomic<bool> workers_running_;
    std::atomic<int> active_tasks_;
    
    // Performance monitoring
    mutable std::mutex stats_mutex_;
    std::unordered_map<OptimizationMethod, size_t> method_usage_;
    std::vector<double> optimization_times_;
    size_t total_optimizations_;
    size_t successful_optimizations_;
    std::vector<double> improvement_rates_;
    rclcpp::Time last_statistics_publish_;
    
    // State
    std::atomic<bool> enabled_;
    std::atomic<bool> initialized_;
    
    // ROS2 interfaces
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr performance_pub_;
    
    rclcpp::Service<motion_planning_srvs::srv::OptimizePath>::SharedPtr optimize_path_service_;
    rclcpp::Service<motion_planning_srvs::srv::OptimizePathBatch>::SharedPtr optimize_path_batch_service_;
    rclcpp::Service<motion_planning_srvs::srv::ComputePathMetrics>::SharedPtr compute_metrics_service_;
    rclcpp::Service<motion_planning_srvs::srv::ValidateTrajectory>::SharedPtr validate_trajectory_service_;
    
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
     * @brief Compute linear interpolation between waypoints
     * 
     * @param start Start waypoint
     * @param end End waypoint
     * @param t Interpolation parameter [0, 1]
     * @param interpolated Output interpolated waypoint
     */
    static void interpolateWaypoints(
        const TrajectoryWaypoint& start,
        const TrajectoryWaypoint& end,
        double t,
        TrajectoryWaypoint& interpolated
    );
    
    /**
     * @brief Compute cubic spline interpolation
     * 
     * @param waypoints Control points
     * @param num_points Number of interpolated points
     * @param interpolated_waypoints Output interpolated waypoints
     * @return true if interpolation successful
     */
    static bool computeCubicSplineInterpolation(
        const std::vector<TrajectoryWaypoint>& waypoints,
        size_t num_points,
        std::vector<TrajectoryWaypoint>& interpolated_waypoints
    );
    
    /**
     * @brief Compute B-spline interpolation
     * 
     * @param waypoints Control points
     * @param degree B-spline degree
     * @param num_points Number of interpolated points
     * @param interpolated_waypoints Output interpolated waypoints
     * @return true if interpolation successful
     */
    static bool computeBSplineInterpolation(
        const std::vector<TrajectoryWaypoint>& waypoints,
        int degree,
        size_t num_points,
        std::vector<TrajectoryWaypoint>& interpolated_waypoints
    );
    
    /**
     * @brief Compute minimum jerk trajectory between waypoints
     * 
     * @param start Start waypoint
     * @param end End waypoint
     * @param duration Trajectory duration
     * @param num_points Number of points
     * @param trajectory Output trajectory waypoints
     * @return true if computation successful
     */
    static bool computeMinimumJerkTrajectory(
        const TrajectoryWaypoint& start,
        const TrajectoryWaypoint& end,
        double duration,
        size_t num_points,
        std::vector<TrajectoryWaypoint>& trajectory
    );
    
    /**
     * @brief Compute trapezoidal velocity profile
     * 
     * @param distance Total distance
     * @param max_velocity Maximum velocity
     * @param max_acceleration Maximum acceleration
     * @param times Output phase times
     * @param velocities Output phase velocities
     * @return true if profile feasible
     */
    static bool computeTrapezoidalVelocityProfile(
        double distance,
        double max_velocity,
        double max_acceleration,
        std::array<double, 3>& times,
        std::array<double, 3>& velocities
    );
    
    /**
     * @brief Compute S-curve velocity profile
     * 
     * @param distance Total distance
     * @param max_velocity Maximum velocity
     * @param max_acceleration Maximum acceleration
     * @param max_jerk Maximum jerk
     * @param times Output phase times
     * @param velocities Output phase velocities
     * @return true if profile feasible
     */
    static bool computeSCurveVelocityProfile(
        double distance,
        double max_velocity,
        double max_acceleration,
        double max_jerk,
        std::array<double, 7>& times,
        std::array<double, 7>& velocities
    );
    
    /**
     * @brief Compute energy consumption for motion segment
     * 
     * @param start Start state
     * @param end End state
     * @param duration Motion duration
     * @param energy Output energy consumption
     * @return true if computation successful
     */
    static bool computeMotionEnergy(
        const TrajectoryWaypoint& start,
        const TrajectoryWaypoint& end,
        double duration,
        double& energy
    );
    
    /**
     * @brief Compute path length in joint space
     * 
     * @param waypoints Trajectory waypoints
     * @param length Output path length
     * @return true if computation successful
     */
    static bool computeJointPathLength(
        const std::vector<TrajectoryWaypoint>& waypoints,
        double& length
    );
    
    /**
     * @brief Compute jerk metric for trajectory
     * 
     * @param waypoints Trajectory waypoints
     * @param total_jerk Output total jerk
     * @param max_jerk Output maximum jerk
     * @return true if computation successful
     */
    static bool computeJerkMetrics(
        const std::vector<TrajectoryWaypoint>& waypoints,
        double& total_jerk,
        double& max_jerk
    );
    
    /**
     * @brief Compute clearance consistency metric
     * 
     * @param clearances Clearance values along path
     * @param consistency Output consistency metric
     * @return true if computation successful
     */
    static bool computeClearanceConsistency(
        const std::vector<double>& clearances,
        double& consistency
    );
};

}  // namespace motion_planning

// Specialize std::hash for CacheKey
namespace std
{
    template<>
    struct hash<motion_planning::PathOptimizer::CacheKey>
    {
        size_t operator()(const motion_planning::PathOptimizer::CacheKey& key) const
        {
            motion_planning::PathOptimizer::CacheKeyHash hasher;
            return hasher(key);
        }
    };
}

#endif  // MOTION_PLANNING__PATH_OPTIMIZER_HPP_
