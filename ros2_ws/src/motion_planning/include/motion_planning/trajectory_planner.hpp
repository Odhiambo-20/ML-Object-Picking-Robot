// ml-object-picking-robot/ros2_ws/src/motion_planning/include/motion_planning/trajectory_planner.hpp
// Industrial-grade Trajectory Planner for Robotic Arm Motion
// Real-time trajectory generation with safety guarantees and optimization

#ifndef MOTION_PLANNING__TRAJECTORY_PLANNER_HPP_
#define MOTION_PLANNING__TRAJECTORY_PLANNER_HPP_

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
#include <map>

// ROS2
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>
#include <trajectory_msgs/msg/joint_trajectory_point.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <nav_msgs/msg/path.hpp>
#include <moveit_msgs/msg/motion_plan_request.hpp>
#include <moveit_msgs/msg/motion_plan_response.hpp>

// OMPL for motion planning
#include <ompl/base/StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/spaces/SE3StateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRT.h>
#include <ompl/geometric/planners/rrt/RRTConnect.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/geometric/planners/prm/PRM.h>
#include <ompl/geometric/planners/prm/PRMstar.h>
#include <ompl/geometric/planners/est/EST.h>
#include <ompl/geometric/planners/kpiece/KPIECE1.h>
#include <ompl/geometric/PathGeometric.h>
#include <ompl/config.h>

// Eigen for advanced mathematics
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Sparse>
#include <Eigen/StdVector>

// Real-time control
#include <control_toolbox/pid.h>
#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

// Thread pooling and async operations
#include <thread>
#include <future>
#include <condition_variable>
#include <chrono>

namespace motion_planning
{

// ============================================
// ENUMS AND CONSTANTS
// ============================================

enum class PlannerType
{
    RRT = 0,                // Rapidly-exploring Random Tree
    RRT_CONNECT = 1,        // RRT-Connect (bidirectional)
    RRT_STAR = 2,           // RRT* (optimal)
    PRM = 3,                // Probabilistic Roadmap
    PRM_STAR = 4,           // PRM* (optimal)
    EST = 5,                // Expansive Space Trees
    KPIECE = 6,             // Kinematic Planning by Interior-Exterior Cell Exploration
    BIT_STAR = 7,           // Batch Informed Trees (optimal)
    FMT = 8,                // Fast Marching Tree
    PDST = 9,               // Path-Directed Subdivision Tree
    HYBRID = 10             // Hybrid planner selection
};

enum class PlanningObjective
{
    PATH_LENGTH = 0,        // Minimize path length
    CLEARANCE = 1,          // Maximize clearance from obstacles
    TIME_OPTIMAL = 2,       // Minimize execution time
    ENERGY_EFFICIENT = 3,   // Minimize energy consumption
    SMOOTHNESS = 4,         // Maximize smoothness
    COMBINED = 5,           // Weighted combination
    SAFETY_FIRST = 6        // Prioritize safety margins
};

enum class PlanningStatus
{
    SUCCESS = 0,
    FAILURE = 1,
    TIMEOUT = 2,
    INVALID_START = 3,
    INVALID_GOAL = 4,
    UNREACHABLE = 5,
    COLLISION_DETECTED = 6,
    JOINT_LIMIT_VIOLATION = 7,
    NUMERICAL_ERROR = 8,
    PLANNER_ERROR = 9
};

enum class TrajectoryType
{
    JOINT_SPACE = 0,        // Joint space trajectory
    CARTESIAN_SPACE = 1,    // Cartesian space trajectory
    MIXED = 2,              // Mixed joint and Cartesian
    VIA_POINTS = 3,         // Via points trajectory
    SPLINE = 4,             // Spline-based trajectory
    ADAPTIVE = 5            // Adaptive trajectory
};

enum class InterpolationMethod
{
    LINEAR = 0,             // Linear interpolation
    CUBIC_SPLINE = 1,       // Cubic spline interpolation
    QUINTIC_SPLINE = 2,     // Quintic spline interpolation
    B_SPLINE = 3,           // B-spline interpolation
    MINIMUM_JERK = 4,       // Minimum jerk trajectory
    TIME_OPTIMAL = 5        // Time-optimal interpolation
};

struct TrajectoryPlannerConfig
{
    // Planner selection
    PlannerType primary_planner;
    std::vector<PlannerType> fallback_planners;
    PlanningObjective objective;
    
    // Planning parameters
    double planning_time;           // Maximum planning time (seconds)
    double simplification_time;     // Time for path simplification
    double validation_resolution;   // Resolution for collision checking
    double goal_tolerance_position; // Position tolerance (meters)
    double goal_tolerance_orientation; // Orientation tolerance (radians)
    
    // OMPL specific parameters
    double range;                   // Maximum step size
    double goal_bias;               // Probability to sample goal
    int max_planning_attempts;      // Maximum planning attempts
    bool enable_simplification;     // Enable path simplification
    bool enable_optimization;       // Enable path optimization
    
    // Trajectory generation
    TrajectoryType trajectory_type;
    InterpolationMethod interpolation_method;
    double interpolation_resolution; // Resolution for interpolation
    bool enable_time_parameterization;
    bool enable_velocity_scaling;
    
    // Time-optimal parameterization
    double max_velocity_scale;      // Scale factor for max velocity
    double max_acceleration_scale;  // Scale factor for max acceleration
    double max_jerk_scale;          // Scale factor for max jerk
    double blending_radius;         // Blending radius for waypoints
    
    // Safety parameters
    double safety_margin;           // Safety margin from obstacles
    double min_clearance;           // Minimum required clearance
    bool enforce_joint_limits;
    bool enforce_velocity_limits;
    bool enforce_torque_limits;
    
    // Real-time parameters
    double control_frequency;       // Control loop frequency (Hz)
    double realtime_tolerance;      // Real-time execution tolerance
    bool enable_preemption;         // Enable trajectory preemption
    bool enable_replanning;         // Enable online replanning
    
    // Performance optimization
    bool enable_caching;
    size_t cache_size;
    double cache_ttl;               // Cache time-to-live (seconds)
    bool enable_parallel_planning;
    int num_threads;
    
    // Monitoring and debugging
    bool enable_statistics;
    bool enable_visualization;
    double statistics_publish_rate;
    bool log_planning_data;
    
    TrajectoryPlannerConfig()
        : primary_planner(PlannerType::RRT_STAR),
          objective(PlanningObjective::COMBINED),
          planning_time(5.0),
          simplification_time(1.0),
          validation_resolution(0.01),      // 1cm
          goal_tolerance_position(0.005),   // 5mm
          goal_tolerance_orientation(0.01), // ~0.57°
          range(0.1),                       // 10cm
          goal_bias(0.05),                  // 5% goal bias
          max_planning_attempts(3),
          enable_simplification(true),
          enable_optimization(true),
          trajectory_type(TrajectoryType::JOINT_SPACE),
          interpolation_method(InterpolationMethod::QUINTIC_SPLINE),
          interpolation_resolution(0.01),   // 1cm
          enable_time_parameterization(true),
          enable_velocity_scaling(true),
          max_velocity_scale(0.8),          // 80% of limits
          max_acceleration_scale(0.7),      // 70% of limits
          max_jerk_scale(0.6),              // 60% of limits
          blending_radius(0.02),            // 2cm
          safety_margin(0.05),              // 5cm
          min_clearance(0.02),              // 2cm
          enforce_joint_limits(true),
          enforce_velocity_limits(true),
          enforce_torque_limits(false),
          control_frequency(125.0),         // 125 Hz (8ms)
          realtime_tolerance(0.001),        // 1ms
          enable_preemption(true),
          enable_replanning(true),
          enable_caching(true),
          cache_size(10000),
          cache_ttl(3600.0),                // 1 hour
          enable_parallel_planning(true),
          num_threads(4),
          enable_statistics(true),
          enable_visualization(false),
          statistics_publish_rate(1.0),
          log_planning_data(true) {}
};

struct PlanningRequest
{
    // Start and goal specifications
    std::vector<double> start_joint_state;
    std::vector<double> goal_joint_state;
    geometry_msgs::msg::Pose start_pose;
    geometry_msgs::msg::Pose goal_pose;
    
    // Planning constraints
    std::vector<geometry_msgs::msg::Pose> collision_objects;
    std::vector<std::string> forbidden_regions;
    std::vector<double> joint_limits_override;
    
    // Planning preferences
    PlannerType preferred_planner;
    PlanningObjective objective;
    double max_planning_time;
    double required_clearance;
    
    // Trajectory preferences
    TrajectoryType trajectory_type;
    bool require_smooth_trajectory;
    bool require_time_optimal;
    double max_trajectory_duration;
    
    // Execution preferences
    bool execute_immediately;
    bool wait_for_completion;
    double execution_velocity_scale;
    
    PlanningRequest()
        : preferred_planner(PlannerType::RRT_STAR),
          objective(PlanningObjective::COMBINED),
          max_planning_time(10.0),
          required_clearance(0.05),
          trajectory_type(TrajectoryType::JOINT_SPACE),
          require_smooth_trajectory(true),
          require_time_optimal(false),
          max_trajectory_duration(30.0),
          execute_immediately(false),
          wait_for_completion(true),
          execution_velocity_scale(1.0) {}
};

struct PlanningResult
{
    trajectory_msgs::msg::JointTrajectory planned_trajectory;
    PlanningStatus status;
    std::string message;
    
    // Planning statistics
    double planning_time;           // Total planning time (seconds)
    double simplification_time;     // Path simplification time
    double optimization_time;       // Path optimization time
    int planning_iterations;
    size_t path_length;             // Number of waypoints
    double path_cost;               // Cost of planned path
    
    // Trajectory metrics
    double trajectory_duration;     // Total execution time
    double trajectory_length;       // Path length in joint space
    double min_clearance;           // Minimum clearance along path
    double average_clearance;       // Average clearance
    double max_velocity;            // Maximum joint velocity
    double max_acceleration;        // Maximum joint acceleration
    
    // Validation results
    bool collision_free;
    bool within_joint_limits;
    bool within_velocity_limits;
    bool smoothness_validated;
    
    // Additional data
    std::vector<geometry_msgs::msg::Pose> cartesian_path;
    std::vector<double> cost_breakdown;
    std::string planner_used;
    
    PlanningResult()
        : status(PlanningStatus::FAILURE),
          planning_time(0.0),
          simplification_time(0.0),
          optimization_time(0.0),
          planning_iterations(0),
          path_length(0),
          path_cost(std::numeric_limits<double>::max()),
          trajectory_duration(0.0),
          trajectory_length(0.0),
          min_clearance(0.0),
          average_clearance(0.0),
          max_velocity(0.0),
          max_acceleration(0.0),
          collision_free(false),
          within_joint_limits(false),
          within_velocity_limits(false),
          smoothness_validated(false) {}
};

struct RealtimePlanningContext
{
    std::vector<double> current_joint_state;
    std::vector<double> target_joint_state;
    geometry_msgs::msg::Pose current_pose;
    geometry_msgs::msg::Pose target_pose;
    
    std::vector<geometry_msgs::msg::Pose> dynamic_obstacles;
    std::vector<double> obstacle_velocities;
    
    double time_horizon;            // Planning horizon (seconds)
    double replanning_interval;     // Replanning interval (seconds)
    double max_deviation;           // Maximum allowed deviation
    
    bool emergency_stop;
    bool pause_requested;
    std::atomic<bool> cancel_requested;
    
    RealtimePlanningContext()
        : time_horizon(5.0),
          replanning_interval(0.1), // 100ms
          max_deviation(0.05),      // 5cm
          emergency_stop(false),
          pause_requested(false),
          cancel_requested(false) {}
};

struct TrajectoryExecutionMonitor
{
    std::vector<double> commanded_positions;
    std::vector<double> actual_positions;
    std::vector<double> position_errors;
    std::vector<double> commanded_velocities;
    std::vector<double> actual_velocities;
    std::vector<double> velocity_errors;
    
    std::vector<double> joint_torques;
    std::vector<double> joint_temperatures;
    std::vector<double> motor_currents;
    
    double execution_progress;      // 0.0 to 1.0
    double time_elapsed;
    double time_remaining;
    double average_tracking_error;
    double max_tracking_error;
    
    bool on_trajectory;
    bool within_tolerance;
    bool safety_violation;
    
    std::vector<std::string> warnings;
    std::vector<std::string> errors;
    
    TrajectoryExecutionMonitor()
        : execution_progress(0.0),
          time_elapsed(0.0),
          time_remaining(0.0),
          average_tracking_error(0.0),
          max_tracking_error(0.0),
          on_trajectory(false),
          within_tolerance(false),
          safety_violation(false) {}
};

// ============================================
// TRAJECTORY PLANNER CLASS
// ============================================

class TrajectoryPlanner : public rclcpp::Node
{
public:
    using SharedPtr = std::shared_ptr<TrajectoryPlanner>;
    using UniquePtr = std::unique_ptr<TrajectoryPlanner>;
    
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    // Action server types
    using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
    using GoalHandleFollowJointTrajectory = rclcpp_action::ServerGoalHandle<FollowJointTrajectory>;
    
    /**
     * @brief Construct a new Trajectory Planner object
     * 
     * @param node_name ROS2 node name
     * @param options ROS2 node options
     */
    explicit TrajectoryPlanner(
        const std::string& node_name = "trajectory_planner",
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions()
    );
    
    /**
     * @brief Destroy the Trajectory Planner object
     */
    ~TrajectoryPlanner();
    
    // ============================================
    // INITIALIZATION AND CONFIGURATION
    // ============================================
    
    /**
     * @brief Initialize the trajectory planner with configuration
     * 
     * @param config Planner configuration
     * @return true if initialization successful
     */
    bool initialize(const TrajectoryPlannerConfig& config);
    
    /**
     * @brief Load robot model for planning
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
     * @brief Set collision checker for planning
     * 
     * @param collision_checker Shared pointer to collision checker
     */
    void setCollisionChecker(std::shared_ptr<class CollisionChecker> collision_checker);
    
    /**
     * @brief Set inverse kinematics solver
     * 
     * @param ik_solver Shared pointer to IK solver
     */
    void setInverseKinematicsSolver(std::shared_ptr<class InverseKinematicsSolver> ik_solver);
    
    /**
     * @brief Set path optimizer
     * 
     * @param path_optimizer Shared pointer to path optimizer
     */
    void setPathOptimizer(std::shared_ptr<class PathOptimizer> path_optimizer);
    
    /**
     * @brief Set joint limits for planning
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
     * @brief Update planner configuration at runtime
     * 
     * @param config New configuration
     */
    void updateConfig(const TrajectoryPlannerConfig& config);
    
    // ============================================
    // PLANNING METHODS
    // ============================================
    
    /**
     * @brief Plan trajectory from start to goal
     * 
     * @param request Planning request
     * @param result Planning result (output)
     * @return true if planning attempted
     */
    bool planTrajectory(const PlanningRequest& request, PlanningResult& result);
    
    /**
     * @brief Plan trajectory (simplified interface)
     * 
     * @param start_joint_state Start joint state
     * @param goal_joint_state Goal joint state
     * @param trajectory Output planned trajectory
     * @param planner_type Planner type to use
     * @return PlanningStatus Planning status
     */
    PlanningStatus planTrajectory(
        const std::vector<double>& start_joint_state,
        const std::vector<double>& goal_joint_state,
        trajectory_msgs::msg::JointTrajectory& trajectory,
        PlannerType planner_type = PlannerType::RRT_STAR
    );
    
    /**
     * @brief Plan Cartesian trajectory
     * 
     * @param start_pose Start pose
     * @param goal_pose Goal pose
     * @param trajectory Output planned trajectory
     * @return PlanningStatus Planning status
     */
    PlanningStatus planCartesianTrajectory(
        const geometry_msgs::msg::Pose& start_pose,
        const geometry_msgs::msg::Pose& goal_pose,
        trajectory_msgs::msg::JointTrajectory& trajectory
    );
    
    /**
     * @brief Plan trajectory through via points
     * 
     * @param via_points Vector of via points (joint states or poses)
     * @param trajectory Output planned trajectory
     * @param via_point_tolerances Tolerances at via points
     * @return PlanningStatus Planning status
     */
    PlanningStatus planViaPointTrajectory(
        const std::vector<std::vector<double>>& via_points,
        trajectory_msgs::msg::JointTrajectory& trajectory,
        const std::vector<double>& via_point_tolerances = {}
    );
    
    /**
     * @brief Replan trajectory from current state
     * 
     * @param original_trajectory Original trajectory
     * @param current_state Current joint state
     * @param replanned_trajectory Output replanned trajectory
     * @param replanning_reason Reason for replanning
     * @return PlanningStatus Planning status
     */
    PlanningStatus replanTrajectory(
        const trajectory_msgs::msg::JointTrajectory& original_trajectory,
        const std::vector<double>& current_state,
        trajectory_msgs::msg::JointTrajectory& replanned_trajectory,
        const std::string& replanning_reason = ""
    );
    
    /**
     * @brief Plan trajectory for pick-and-place operation
     * 
     * @param pick_pose Pick pose
     * @param place_pose Place pose
     * @param approach_distance Approach distance
     * @param retreat_distance Retreat distance
     * @param trajectory Output planned trajectory
     * @return PlanningStatus Planning status
     */
    PlanningStatus planPickPlaceTrajectory(
        const geometry_msgs::msg::Pose& pick_pose,
        const geometry_msgs::msg::Pose& place_pose,
        double approach_distance,
        double retreat_distance,
        trajectory_msgs::msg::JointTrajectory& trajectory
    );
    
    // ============================================
    // REAL-TIME PLANNING AND EXECUTION
    // ============================================
    
    /**
     * @brief Start real-time planning for given context
     * 
     * @param context Real-time planning context
     * @return true if real-time planning started
     */
    bool startRealtimePlanning(const RealtimePlanningContext& context);
    
    /**
     * @brief Stop real-time planning
     */
    void stopRealtimePlanning();
    
    /**
     * @brief Execute trajectory with real-time monitoring
     * 
     * @param trajectory Trajectory to execute
     * @param monitor Trajectory execution monitor (output)
     * @return true if execution started
     */
    bool executeTrajectory(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        TrajectoryExecutionMonitor& monitor
    );
    
    /**
     * @brief Stop trajectory execution
     * 
     * @param emergency True for emergency stop
     */
    void stopExecution(bool emergency = false);
    
    /**
     * @brief Pause trajectory execution
     */
    void pauseExecution();
    
    /**
     * @brief Resume trajectory execution
     */
    void resumeExecution();
    
    /**
     * @brief Get current execution status
     * 
     * @return TrajectoryExecutionMonitor Current execution status
     */
    TrajectoryExecutionMonitor getExecutionStatus() const;
    
    // ============================================
    // TRAJECTORY GENERATION AND PROCESSING
    // ============================================
    
    /**
     * @brief Generate smooth trajectory from path
     * 
     * @param path Joint space path (positions only)
     * @param trajectory Output smooth trajectory
     * @param method Interpolation method
     * @return true if generation successful
     */
    bool generateSmoothTrajectory(
        const std::vector<std::vector<double>>& path,
        trajectory_msgs::msg::JointTrajectory& trajectory,
        InterpolationMethod method = InterpolationMethod::QUINTIC_SPLINE
    );
    
    /**
     * @brief Apply time-optimal parameterization to trajectory
     * 
     * @param trajectory Input trajectory
     * @param time_parameterized_trajectory Output time-parameterized trajectory
     * @return true if parameterization successful
     */
    bool applyTimeOptimalParameterization(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        trajectory_msgs::msg::JointTrajectory& time_parameterized_trajectory
    );
    
    /**
     * @brief Blend two trajectories
     * 
     * @param trajectory1 First trajectory
     * @param trajectory2 Second trajectory
     * @param blend_point Blend point (0.0 to 1.0)
     * @param blended_trajectory Output blended trajectory
     * @return true if blending successful
     */
    bool blendTrajectories(
        const trajectory_msgs::msg::JointTrajectory& trajectory1,
        const trajectory_msgs::msg::JointTrajectory& trajectory2,
        double blend_point,
        trajectory_msgs::msg::JointTrajectory& blended_trajectory
    );
    
    /**
     * @brief Scale trajectory velocity
     * 
     * @param trajectory Input trajectory
     * @param scale_factor Velocity scale factor (0.0 to 1.0)
     * @param scaled_trajectory Output scaled trajectory
     * @return true if scaling successful
     */
    bool scaleTrajectoryVelocity(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        double scale_factor,
        trajectory_msgs::msg::JointTrajectory& scaled_trajectory
    );
    
    /**
     * @brief Reverse trajectory
     * 
     * @param trajectory Input trajectory
     * @param reversed_trajectory Output reversed trajectory
     * @return true if reversal successful
     */
    bool reverseTrajectory(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        trajectory_msgs::msg::JointTrajectory& reversed_trajectory
    );
    
    // ============================================
    // VALIDATION AND ANALYSIS
    // ============================================
    
    /**
     * @brief Validate trajectory for execution
     * 
     * @param trajectory Trajectory to validate
     * @param validation_result Output validation result
     * @return true if trajectory is valid
     */
    bool validateTrajectory(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        PlanningResult& validation_result
    );
    
    /**
     * @brief Check trajectory for collisions
     * 
     * @param trajectory Trajectory to check
     * @param collision_points Output collision points
     * @return true if collision-free
     */
    bool checkTrajectoryCollisions(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        std::vector<geometry_msgs::msg::Pose>& collision_points
    );
    
    /**
     * @brief Compute trajectory metrics
     * 
     * @param trajectory Trajectory to analyze
     * @param metrics Output trajectory metrics
     * @return true if computation successful
     */
    bool computeTrajectoryMetrics(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        PlanningResult& metrics
    );
    
    /**
     * @brief Compare two trajectories
     * 
     * @param trajectory1 First trajectory
     * @param trajectory2 Second trajectory
     * @param comparison_result Output comparison result
     * @return true if comparison successful
     */
    bool compareTrajectories(
        const trajectory_msgs::msg::JointTrajectory& trajectory1,
        const trajectory_msgs::msg::JointTrajectory& trajectory2,
        std::unordered_map<std::string, double>& comparison_result
    );
    
    // ============================================
    // CACHING AND PERFORMANCE
    // ============================================
    
    /**
     * @brief Clear planning cache
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
     * @brief Get planning performance statistics
     * 
     * @param avg_planning_time_ms Average planning time
     * @param success_rate Success rate percentage
     * @param avg_path_length Average path length
     */
    void getPerformanceStatistics(
        double& avg_planning_time_ms,
        double& success_rate,
        double& avg_path_length
    ) const;
    
    /**
     * @brief Reset performance statistics
     */
    void resetStatistics();
    
    // ============================================
    // VISUALIZATION AND DEBUGGING
    // ============================================
    
    /**
     * @brief Get planning visualization markers
     * 
     * @param result Planning result
     * @param marker_array Output marker array
     */
    void getPlanningMarkers(
        const PlanningResult& result,
        visualization_msgs::msg::MarkerArray& marker_array
    );
    
    /**
     * @brief Get trajectory visualization markers
     * 
     * @param trajectory Trajectory to visualize
     * @param marker_array Output marker array
     * @param color Color for visualization
     */
    void getTrajectoryMarkers(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        visualization_msgs::msg::MarkerArray& marker_array,
        const std::array<double, 4>& color = {0.0, 1.0, 0.0, 1.0}
    );
    
    /**
     * @brief Get execution visualization markers
     * 
     * @param monitor Execution monitor
     * @param marker_array Output marker array
     */
    void getExecutionMarkers(
        const TrajectoryExecutionMonitor& monitor,
        visualization_msgs::msg::MarkerArray& marker_array
    );
    
    // ============================================
    // CONFIGURATION AND STATE
    // ============================================
    
    /**
     * @brief Get current planner configuration
     * 
     * @return TrajectoryPlannerConfig Current configuration
     */
    TrajectoryPlannerConfig getConfig() const;
    
    /**
     * @brief Check if planner is initialized
     * 
     * @return true if initialized
     */
    bool isInitialized() const;
    
    /**
     * @brief Enable/disable planner
     * 
     * @param enabled True to enable
     */
    void setEnabled(bool enabled);
    
    /**
     * @brief Check if planner is enabled
     * 
     * @return true if enabled
     */
    bool isEnabled() const;
    
    /**
     * @brief Check if real-time planning is active
     * 
     * @return true if real-time planning active
     */
    bool isRealtimePlanningActive() const;
    
    /**
     * @brief Check if execution is active
     * 
     * @return true if execution active
     */
    bool isExecutionActive() const;

private:
    // ============================================
    // PRIVATE TYPES
    // ============================================
    
    struct CacheKey
    {
        std::vector<double> start_state;
        std::vector<double> goal_state;
        PlannerType planner_type;
        PlanningObjective objective;
        std::string constraints_hash;
        
        bool operator==(const CacheKey& other) const;
    };
    
    struct CacheKeyHash
    {
        size_t operator()(const CacheKey& key) const;
    };
    
    struct CacheEntry
    {
        PlanningResult result;
        rclcpp::Time timestamp;
        size_t access_count;
        double average_access_time;
        
        CacheEntry() : access_count(0), average_access_time(0.0) {}
    };
    
    struct PlanningTask
    {
        PlanningRequest request;
        std::promise<PlanningResult> promise;
        rclcpp::Time submission_time;
        
        PlanningTask(PlanningRequest req, std::promise<PlanningResult> prom)
            : request(std::move(req)), promise(std::move(prom)) {}
    };
    
    struct OMPLStateSpaceWrapper
    {
        std::shared_ptr<ompl::base::RealVectorStateSpace> state_space;
        std::shared_ptr<ompl::base::SpaceInformation> space_info;
        std::shared_ptr<ompl::base::ProblemDefinition> problem_def;
        std::shared_ptr<ompl::geometric::SimpleSetup> simple_setup;
        
        OMPLStateSpaceWrapper() {}
    };
    
    // ============================================
    // PRIVATE METHODS
    // ============================================
    
    /**
     * @brief Initialize OMPL planners
     */
    void initializeOMPL();
    
    /**
     * @brief Initialize ROS2 action servers
     */
    void initializeActionServers();
    
    /**
     * @brief Main planning method with planner selection
     * 
     * @param request Planning request
     * @param result Planning result
     * @return true if planning attempted
     */
    bool planTrajectoryInternal(const PlanningRequest& request, PlanningResult& result);
    
    /**
     * @brief Plan using OMPL
     * 
     * @param request Planning request
     * @param result Planning result
     * @return PlanningStatus Planning status
     */
    PlanningStatus planWithOMPL(const PlanningRequest& request, PlanningResult& result);
    
    /**
     * @brief Create OMPL state space for robot
     * 
     * @param state_space Output state space wrapper
     * @return true if creation successful
     */
    bool createOMPLStateSpace(OMPLStateSpaceWrapper& state_space);
    
    /**
     * @brief Create OMPL planner instance
     * 
     * @param space_info OMPL space information
     * @param planner_type Type of planner to create
     * @return std::shared_ptr<ompl::base::Planner> Planner instance
     */
    std::shared_ptr<ompl::base::Planner> createOMPLPlanner(
        const ompl::base::SpaceInformationPtr& space_info,
        PlannerType planner_type
    );
    
    /**
     * @brief Convert OMPL path to joint trajectory
     * 
     * @param ompl_path OMPL geometric path
     * @param trajectory Output joint trajectory
     * @return true if conversion successful
     */
    bool convertOMPLPathToTrajectory(
        const ompl::geometric::PathGeometric& ompl_path,
        trajectory_msgs::msg::JointTrajectory& trajectory
    );
    
    /**
     * @brief Plan Cartesian path using IK
     * 
     * @param start_pose Start pose
     * @param goal_pose Goal pose
     * @param trajectory Output trajectory
     * @return PlanningStatus Planning status
     */
    PlanningStatus planCartesianPathInternal(
        const geometry_msgs::msg::Pose& start_pose,
        const geometry_msgs::msg::Pose& goal_pose,
        trajectory_msgs::msg::JointTrajectory& trajectory
    );
    
    /**
     * @brief Generate via point trajectory
     * 
     * @param via_points Via points
     * @param trajectory Output trajectory
     * @return PlanningStatus Planning status
     */
    PlanningStatus generateViaPointTrajectory(
        const std::vector<std::vector<double>>& via_points,
        trajectory_msgs::msg::JointTrajectory& trajectory
    );
    
    /**
     * @brief Check cache for existing plan
     * 
     * @param key Cache key
     * @param result Output result if found
     * @return true if cache hit
     */
    bool checkCache(const CacheKey& key, PlanningResult& result);
    
    /**
     * @brief Update cache with new plan
     * 
     * @param key Cache key
     * @param result Result to cache
     */
    void updateCache(const CacheKey& key, const PlanningResult& result);
    
    /**
     * @brief Clean expired cache entries
     */
    void cleanCache();
    
    /**
     * @brief Real-time planning thread function
     */
    void realtimePlanningThread();
    
    /**
     * @brief Execution monitoring thread function
     */
    void executionMonitoringThread();
    
    /**
     * @brief Process planning task
     * 
     * @param task Planning task
     */
    void processPlanningTask(std::shared_ptr<PlanningTask> task);
    
    /**
     * @brief Generate smooth trajectory between points
     * 
     * @param start_point Start point
     * @param end_point End point
     * @param duration Trajectory duration
     * @param num_points Number of points
     * @param method Interpolation method
     * @param trajectory Output trajectory segment
     * @return true if generation successful
     */
    bool generateTrajectorySegment(
        const trajectory_msgs::msg::JointTrajectoryPoint& start_point,
        const trajectory_msgs::msg::JointTrajectoryPoint& end_point,
        double duration,
        size_t num_points,
        InterpolationMethod method,
        trajectory_msgs::msg::JointTrajectory& trajectory
    );
    
    /**
     * @brief Apply quintic spline interpolation
     * 
     * @param start Start point
     * @param end End point
     * @param duration Duration
     * @param num_points Number of points
     * @param trajectory Output trajectory
     * @return true if interpolation successful
     */
    bool applyQuinticSplineInterpolation(
        const trajectory_msgs::msg::JointTrajectoryPoint& start,
        const trajectory_msgs::msg::JointTrajectoryPoint& end,
        double duration,
        size_t num_points,
        trajectory_msgs::msg::JointTrajectory& trajectory
    );
    
    /**
     * @brief Apply cubic spline interpolation
     * 
     * @param waypoints Control points
     * @param trajectory Output trajectory
     * @return true if interpolation successful
     */
    bool applyCubicSplineInterpolation(
        const std::vector<trajectory_msgs::msg::JointTrajectoryPoint>& waypoints,
        trajectory_msgs::msg::JointTrajectory& trajectory
    );
    
    /**
     * @brief Apply B-spline interpolation
     * 
     * @param waypoints Control points
     * @param degree B-spline degree
     * @param trajectory Output trajectory
     * @return true if interpolation successful
     */
    bool applyBSplineInterpolation(
        const std::vector<trajectory_msgs::msg::JointTrajectoryPoint>& waypoints,
        int degree,
        trajectory_msgs::msg::JointTrajectory& trajectory
    );
    
    /**
     * @brief Compute time-optimal parameterization using trapezoidal profile
     * 
     * @param trajectory Input trajectory
     * @param parameterized_trajectory Output parameterized trajectory
     * @return true if parameterization successful
     */
    bool computeTrapezoidalTimeParameterization(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        trajectory_msgs::msg::JointTrajectory& parameterized_trajectory
    );
    
    /**
     * @brief Compute time-optimal parameterization using S-curve profile
     * 
     * @param trajectory Input trajectory
     * @param parameterized_trajectory Output parameterized trajectory
     * @return true if parameterization successful
     */
    bool computeSCurveTimeParameterization(
        const trajectory_msgs::msg::JointTrajectory& trajectory,
        trajectory_msgs::msg::JointTrajectory& parameterized_trajectory
    );
    
    /**
     * @brief Action server goal callback
     * 
     * @param goal_handle Action goal handle
     */
    void handleActionGoal(
        const std::shared_ptr<GoalHandleFollowJointTrajectory> goal_handle
    );
    
    /**
     * @brief Action server cancel callback
     * 
     * @param goal_handle Action goal handle
     */
    void handleActionCancel(
        const std::shared_ptr<GoalHandleFollowJointTrajectory> goal_handle
    );
    
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
    TrajectoryPlannerConfig config_;
    std::shared_mutex config_mutex_;
    
    // Dependencies
    std::shared_ptr<class CollisionChecker> collision_checker_;
    std::shared_ptr<class InverseKinematicsSolver> ik_solver_;
    std::shared_ptr<class PathOptimizer> path_optimizer_;
    
    // Robot model
    std::shared_ptr<urdf::Model> urdf_model_;
    std::shared_ptr<srdf::Model> srdf_model_;
    std::vector<std::string> joint_names_;
    size_t num_joints_;
    
    // Joint limits
    std::vector<double> joint_lower_limits_;
    std::vector<double> joint_upper_limits_;
    std::vector<double> joint_velocity_limits_;
    std::vector<double> joint_acceleration_limits_;
    std::vector<double> joint_jerk_limits_;
    
    // OMPL components
    OMPLStateSpaceWrapper ompl_state_space_;
    std::unordered_map<PlannerType, std::shared_ptr<ompl::base::Planner>> ompl_planners_;
    
    // Cache
    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> planning_cache_;
    std::mutex cache_mutex_;
    size_t cache_hits_;
    size_t cache_misses_;
    
    // Thread pool for planning
    std::vector<std::thread> planning_threads_;
    std::queue<std::shared_ptr<PlanningTask>> planning_queue_;
    std::mutex planning_queue_mutex_;
    std::condition_variable planning_queue_condition_;
    std::atomic<bool> planning_workers_running_;
    std::atomic<int> active_planning_tasks_;
    
    // Real-time planning
    std::thread realtime_planning_thread_;
    std::atomic<bool> realtime_planning_active_;
    RealtimePlanningContext realtime_context_;
    std::shared_mutex realtime_context_mutex_;
    
    // Execution monitoring
    std::thread execution_monitor_thread_;
    std::atomic<bool> execution_monitor_active_;
    TrajectoryExecutionMonitor execution_monitor_;
    std::shared_mutex execution_monitor_mutex_;
    
    // Current execution state
    trajectory_msgs::msg::JointTrajectory current_trajectory_;
    std::atomic<bool> execution_active_;
    std::atomic<bool> execution_paused_;
    std::atomic<bool> emergency_stop_;
    rclcpp::Time execution_start_time_;
    
    // PID controllers for execution monitoring
    std::vector<control_toolbox::Pid> joint_pid_controllers_;
    
    // Performance monitoring
    mutable std::mutex stats_mutex_;
    std::unordered_map<PlannerType, size_t> planner_usage_;
    std::vector<double> planning_times_;
    size_t total_plans_;
    size_t successful_plans_;
    std::vector<double> trajectory_durations_;
    rclcpp::Time last_statistics_publish_;
    
    // State
    std::atomic<bool> enabled_;
    std::atomic<bool> initialized_;
    
    // ROS2 interfaces
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr performance_pub_;
    rclcpp::Publisher<trajectory_msgs::msg::JointTrajectory>::SharedPtr trajectory_pub_;
    
    rclcpp::Service<motion_planning_srvs::srv::PlanTrajectory>::SharedPtr plan_trajectory_service_;
    rclcpp::Service<motion_planning_srvs::srv::PlanCartesianTrajectory>::SharedPtr plan_cartesian_service_;
    rclcpp::Service<motion_planning_srvs::srv::ValidateTrajectory>::SharedPtr validate_trajectory_service_;
    rclcpp::Service<motion_planning_srvs::srv::ExecuteTrajectory>::SharedPtr execute_trajectory_service_;
    
    // Action server
    rclcpp_action::Server<FollowJointTrajectory>::SharedPtr follow_joint_trajectory_action_server_;
    
    // Timer for periodic tasks
    rclcpp::TimerBase::SharedPtr statistics_timer_;
    rclcpp::TimerBase::SharedPtr visualization_timer_;
    rclcpp::TimerBase::SharedPtr cache_cleanup_timer_;
    rclcpp::TimerBase::SharedPtr execution_monitor_timer_;
    
    // Callback groups for parallel execution
    rclcpp::CallbackGroup::SharedPtr planning_callback_group_;
    rclcpp::CallbackGroup::SharedPtr execution_callback_group_;
    rclcpp::CallbackGroup::SharedPtr action_callback_group_;
    
    // ============================================
    // PRIVATE STATIC METHODS
    // ============================================
    
    /**
     * @brief Convert joint state to OMPL state
     * 
     * @param joint_state Joint state
     * @param ompl_state Output OMPL state
     */
    static void jointStateToOMPLState(
        const std::vector<double>& joint_state,
        ompl::base::ScopedState<ompl::base::RealVectorStateSpace>& ompl_state
    );
    
    /**
     * @brief Convert OMPL state to joint state
     * 
     * @param ompl_state OMPL state
     * @param joint_state Output joint state
     */
    static void omplStateToJointState(
        const ompl::base::ScopedState<ompl::base::RealVectorStateSpace>& ompl_state,
        std::vector<double>& joint_state
    );
    
    /**
     * @brief Compute distance between joint states
     * 
     * @param state1 First joint state
     * @param state2 Second joint state
     * @return double Distance
     */
    static double computeJointStateDistance(
        const std::vector<double>& state1,
        const std::vector<double>& state2
    );
    
    /**
     * @brief Compute path length in joint space
     * 
     * @param path Sequence of joint states
     * @return double Path length
     */
    static double computeJointPathLength(
        const std::vector<std::vector<double>>& path
    );
    
    /**
     * @brief Compute clearance for joint state
     * 
     * @param joint_state Joint state
     * @param collision_checker Collision checker
     * @return double Clearance value
     */
    static double computeJointStateClearance(
        const std::vector<double>& joint_state,
        const std::shared_ptr<class CollisionChecker>& collision_checker
    );
    
    /**
     * @brief Interpolate between joint states
     * 
     * @param start Start state
     * @param end End state
     * @param t Interpolation parameter [0, 1]
     * @param interpolated Output interpolated state
     */
    static void interpolateJointStates(
        const std::vector<double>& start,
        const std::vector<double>& end,
        double t,
        std::vector<double>& interpolated
    );
    
    /**
     * @brief Compute quintic polynomial coefficients
     * 
     * @param start_pos Start position
     * @param end_pos End position
     * @param start_vel Start velocity
     * @param end_vel End velocity
     * @param start_acc Start acceleration
     * @param end_acc End acceleration
     * @param duration Duration
     * @return std::array<double, 6> Polynomial coefficients
     */
    static std::array<double, 6> computeQuinticPolynomialCoefficients(
        double start_pos,
        double end_pos,
        double start_vel,
        double end_vel,
        double start_acc,
        double end_acc,
        double duration
    );
    
    /**
     * @brief Compute cubic polynomial coefficients
     * 
     * @param start_pos Start position
     * @param end_pos End position
     * @param start_vel Start velocity
     * @param end_vel End velocity
     * @param duration Duration
     * @return std::array<double, 4> Polynomial coefficients
     */
    static std::array<double, 4> computeCubicPolynomialCoefficients(
        double start_pos,
        double end_pos,
        double start_vel,
        double end_vel,
        double duration
    );
    
    /**
     * @brief Evaluate polynomial
     * 
     * @param coeffs Polynomial coefficients
     * @param t Time parameter
     * @param order Derivative order (0=position, 1=velocity, 2=acceleration)
     * @return double Evaluated value
     */
    static double evaluatePolynomial(
        const std::array<double, 6>& coeffs,
        double t,
        int order = 0
    );
    
    /**
     * @brief Compute trapezoidal velocity profile times
     * 
     * @param distance Total distance
     * @param max_vel Maximum velocity
     * @param max_acc Maximum acceleration
     * @param times Output phase times
     * @return true if profile feasible
     */
    static bool computeTrapezoidalProfileTimes(
        double distance,
        double max_vel,
        double max_acc,
        std::array<double, 3>& times
    );
    
    /**
     * @brief Compute S-curve velocity profile times
     * 
     * @param distance Total distance
     * @param max_vel Maximum velocity
     * @param max_acc Maximum acceleration
     * @param max_jerk Maximum jerk
     * @param times Output phase times
     * @return true if profile feasible
     */
    static bool computeSCurveProfileTimes(
        double distance,
        double max_vel,
        double max_acc,
        double max_jerk,
        std::array<double, 7>& times
    );
};

}  // namespace motion_planning

// Specialize std::hash for CacheKey
namespace std
{
    template<>
    struct hash<motion_planning::TrajectoryPlanner::CacheKey>
    {
        size_t operator()(const motion_planning::TrajectoryPlanner::CacheKey& key) const
        {
            motion_planning::TrajectoryPlanner::CacheKeyHash hasher;
            return hasher(key);
        }
    };
}

#endif  // MOTION_PLANNING__TRAJECTORY_PLANNER_HPP_
