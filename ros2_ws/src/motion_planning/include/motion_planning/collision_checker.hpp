// ml-object-picking-robot/ros2_ws/src/motion_planning/include/motion_planning/collision_checker.hpp
// Industrial-grade Collision Checker for Robotic Arm Motion Planning
// ISO 10218-1:2011 compliant with continuous collision detection

#ifndef MOTION_PLANNING__COLLISION_CHECKER_HPP_
#define MOTION_PLANNING__COLLISION_CHECKER_HPP_

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <shared_mutex>
#include <functional>

// ROS2
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <octomap_msgs/msg/octomap.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

// FCL (Flexible Collision Library)
#include <fcl/narrowphase/collision.h>
#include <fcl/narrowphase/distance.h>
#include <fcl/broadphase/broadphase_bruteforce.h>
#include <fcl/broadphase/broadphase_spatialhash.h>
#include <fcl/broadphase/broadphase_interval_tree.h>
#include <fcl/geometry/shape/box.h>
#include <fcl/geometry/shape/cylinder.h>
#include <fcl/geometry/shape/sphere.h>
#include <fcl/geometry/shape/convex.h>
#include <fcl/geometry/bvh/BVH_model.h>
#include <fcl/geometry/octree/octree.h>

// KDL for kinematics
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/frames.hpp>

// Thread pooling
#include <thread>
#include <future>
#include <queue>
#include <condition_variable>

namespace motion_planning
{

// ============================================
// ENUMS AND CONSTANTS
// ============================================

enum class CollisionCheckerType
{
    FCL = 0,
    BULLET = 1,
    PQP = 2,
    CUSTOM = 3
};

enum class BroadphaseType
{
    BRUTE_FORCE = 0,
    SPATIAL_HASH = 1,
    INTERVAL_TREE = 2,
    DYNAMIC_AABB_TREE = 3
};

enum class NarrowphaseType
{
    GJK = 0,
    EPA = 1,
    GJK_EPA = 2
};

enum class CollisionStatus
{
    COLLISION_FREE = 0,
    IN_COLLISION = 1,
    WITHIN_WARNING_DISTANCE = 2,
    WITHIN_SAFETY_MARGIN = 3,
    UNKNOWN = 4
};

enum class CollisionResponse
{
    NO_RESPONSE = 0,
    EMERGENCY_STOP = 1,
    REDUCE_SPEED = 2,
    REJECT_TRAJECTORY = 3,
    WARNING_ONLY = 4,
    PAUSE_AND_REPLAN = 5
};

struct SafetyZone
{
    std::string name;
    std::string type;  // "box", "sphere", "cylinder", "mesh"
    Eigen::Vector3d dimensions;
    Eigen::Vector3d position;
    Eigen::Quaterniond orientation;
    double padding;
    int priority;
    CollisionResponse response;
    
    // Time-based restrictions
    rclcpp::Time active_start;
    rclcpp::Time active_end;
    bool is_active;
    
    SafetyZone() 
        : padding(0.05), priority(50), response(CollisionResponse::WARNING_ONLY),
          is_active(true) {}
};

struct CollisionCheckResult
{
    bool in_collision;
    double min_distance;
    std::vector<std::pair<std::string, std::string>> colliding_pairs;
    std::vector<double> penetration_depths;
    std::vector<Eigen::Vector3d> collision_points;
    std::vector<Eigen::Vector3d> collision_normals;
    rclcpp::Time check_time;
    double computation_time_ms;
    
    // Safety metrics
    double clearance_to_obstacle;
    double time_to_collision;
    Eigen::Vector3d closest_point_on_robot;
    Eigen::Vector3d closest_point_on_obstacle;
    
    CollisionCheckResult()
        : in_collision(false), min_distance(std::numeric_limits<double>::max()),
          computation_time_ms(0.0), clearance_to_obstacle(std::numeric_limits<double>::max()),
          time_to_collision(std::numeric_limits<double>::max()) {}
};

struct ContinuousCollisionCheckResult : public CollisionCheckResult
{
    double time_of_collision;  // Normalized time [0, 1]
    Eigen::Vector3d collision_velocity;
    double collision_energy;
    
    ContinuousCollisionCheckResult()
        : time_of_collision(1.0), collision_energy(0.0) {}
};

struct CollisionObject
{
    std::string name;
    std::shared_ptr<fcl::CollisionGeometryd> geometry;
    fcl::Transform3d transform;
    std::string type;  // "robot", "environment", "dynamic", "safety"
    bool is_static;
    int collision_group;
    int collision_mask;
    std::vector<double> metadata;  // For custom properties
    
    CollisionObject() 
        : is_static(true), collision_group(1), collision_mask(0xFFFFFFFF) {}
};

struct CollisionCheckerConfig
{
    // Performance settings
    double max_check_time_ms;
    int max_collision_pairs;
    bool enable_caching;
    size_t cache_size;
    double cache_ttl_seconds;
    
    // Safety margins
    double warning_distance;
    double safety_margin;
    double emergency_stop_distance;
    
    // Broadphase settings
    BroadphaseType broadphase_type;
    double spatial_hash_resolution;
    int spatial_hash_table_size;
    
    // Narrowphase settings
    NarrowphaseType narrowphase_type;
    double collision_tolerance;
    double distance_tolerance;
    int max_gjk_iterations;
    int max_epa_iterations;
    
    // Continuous collision detection
    bool enable_ccd;
    double ccd_resolution;
    double ccd_max_penetration;
    bool enable_signed_distance;
    
    // Multi-threading
    bool enable_parallel_checking;
    int num_threads;
    int task_batch_size;
    
    // Monitoring
    bool enable_statistics;
    size_t statistics_window_size;
    double statistics_publish_rate;
    
    CollisionCheckerConfig()
        : max_check_time_ms(10.0),
          max_collision_pairs(100),
          enable_caching(true),
          cache_size(10000),
          cache_ttl_seconds(3600.0),
          warning_distance(0.2),
          safety_margin(0.1),
          emergency_stop_distance(0.05),
          broadphase_type(BroadphaseType::SPATIAL_HASH),
          spatial_hash_resolution(0.1),
          spatial_hash_table_size(1000),
          narrowphase_type(NarrowphaseType::GJK_EPA),
          collision_tolerance(0.001),
          distance_tolerance(0.0001),
          max_gjk_iterations(1000),
          max_epa_iterations(100),
          enable_ccd(true),
          ccd_resolution(0.01),
          ccd_max_penetration(0.02),
          enable_signed_distance(true),
          enable_parallel_checking(true),
          num_threads(4),
          task_batch_size(10),
          enable_statistics(true),
          statistics_window_size(1000),
          statistics_publish_rate(1.0) {}
};

// ============================================
// COLLISION CHECKER CLASS
// ============================================

class CollisionChecker : public rclcpp::Node
{
public:
    using SharedPtr = std::shared_ptr<CollisionChecker>;
    using UniquePtr = std::unique_ptr<CollisionChecker>;
    
    /**
     * @brief Construct a new Collision Checker object
     * 
     * @param node_name ROS2 node name
     * @param options ROS2 node options
     */
    explicit CollisionChecker(
        const std::string& node_name = "collision_checker",
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions()
    );
    
    /**
     * @brief Destroy the Collision Checker object
     */
    ~CollisionChecker();
    
    // ============================================
    // INITIALIZATION AND CONFIGURATION
    // ============================================
    
    /**
     * @brief Initialize the collision checker with configuration
     * 
     * @param config Configuration parameters
     * @return true if initialization successful
     */
    bool initialize(const CollisionCheckerConfig& config);
    
    /**
     * @brief Load robot model for collision checking
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
     * @brief Add environment collision objects
     * 
     * @param objects List of collision objects
     */
    void addEnvironmentObjects(const std::vector<CollisionObject>& objects);
    
    /**
     * @brief Add dynamic obstacle
     * 
     * @param object Dynamic collision object
     */
    void addDynamicObstacle(const CollisionObject& object);
    
    /**
     * @brief Update dynamic obstacle pose
     * 
     * @param name Obstacle name
     * @param transform New transform
     */
    void updateDynamicObstacle(
        const std::string& name,
        const fcl::Transform3d& transform
    );
    
    /**
     * @brief Remove collision object
     * 
     * @param name Object name to remove
     */
    void removeCollisionObject(const std::string& name);
    
    /**
     * @brief Clear all environment objects
     */
    void clearEnvironmentObjects();
    
    /**
     * @brief Define safety zone
     * 
     * @param zone Safety zone definition
     */
    void addSafetyZone(const SafetyZone& zone);
    
    // ============================================
    // COLLISION CHECKING METHODS
    // ============================================
    
    /**
     * @brief Check collision for given joint state
     * 
     * @param joint_state Current joint positions
     * @param result Collision check result (output)
     * @return true if check completed successfully
     */
    bool checkCollision(
        const sensor_msgs::msg::JointState& joint_state,
        CollisionCheckResult& result
    );
    
    /**
     * @brief Check collision for given robot pose
     * 
     * @param robot_pose Robot pose in world frame
     * @param result Collision check result (output)
     * @return true if check completed successfully
     */
    bool checkCollision(
        const geometry_msgs::msg::Pose& robot_pose,
        CollisionCheckResult& result
    );
    
    /**
     * @brief Check collision between two sets of objects
     * 
     * @param group1 First group of objects
     * @param group2 Second group of objects
     * @param result Collision check result (output)
     * @return true if check completed successfully
     */
    bool checkCollisionBetweenGroups(
        const std::vector<std::string>& group1,
        const std::vector<std::string>& group2,
        CollisionCheckResult& result
    );
    
    /**
     * @brief Continuous collision check between two states
     * 
     * @param start_state Start joint state
     * @param end_state End joint state
     * @param result Continuous collision result (output)
     * @return true if check completed successfully
     */
    bool checkContinuousCollision(
        const sensor_msgs::msg::JointState& start_state,
        const sensor_msgs::msg::JointState& end_state,
        ContinuousCollisionCheckResult& result
    );
    
    /**
     * @brief Check if path is collision-free
     * 
     * @param trajectory Sequence of joint states
     * @param resolution Discretization resolution
     * @param results Vector of collision results (output)
     * @return true if entire path is collision-free
     */
    bool checkPathCollision(
        const std::vector<sensor_msgs::msg::JointState>& trajectory,
        double resolution,
        std::vector<CollisionCheckResult>& results
    );
    
    /**
     * @brief Compute signed distance field
     * 
     * @param query_points Points to compute distance at
     * @param distances Output distances (negative for penetration)
     * @param gradients Output distance gradients
     * @return true if computation successful
     */
    bool computeSignedDistanceField(
        const std::vector<Eigen::Vector3d>& query_points,
        std::vector<double>& distances,
        std::vector<Eigen::Vector3d>& gradients
    );
    
    /**
     * @brief Check self-collision of robot
     * 
     * @param joint_state Joint state to check
     * @param result Collision check result (output)
     * @return true if check completed
     */
    bool checkSelfCollision(
        const sensor_msgs::msg::JointState& joint_state,
        CollisionCheckResult& result
    );
    
    /**
     * @brief Check environment collision
     * 
     * @param joint_state Joint state to check
     * @param result Collision check result (output)
     * @return true if check completed
     */
    bool checkEnvironmentCollision(
        const sensor_msgs::msg::JointState& joint_state,
        CollisionCheckResult& result
    );
    
    // ============================================
    // DISTANCE QUERIES
    // ============================================
    
    /**
     * @brief Compute minimum distance to obstacles
     * 
     * @param joint_state Joint state
     * @return double Minimum distance (negative for penetration)
     */
    double computeMinimumDistance(
        const sensor_msgs::msg::JointState& joint_state
    );
    
    /**
     * @brief Compute distance between two objects
     * 
     * @param obj1_name First object name
     * @param obj2_name Second object name
     * @param distance Output distance
     * @param p1 Output closest point on first object
     * @param p2 Output closest point on second object
     * @return true if computation successful
     */
    bool computeDistanceBetweenObjects(
        const std::string& obj1_name,
        const std::string& obj2_name,
        double& distance,
        Eigen::Vector3d& p1,
        Eigen::Vector3d& p2
    );
    
    /**
     * @brief Compute clearance map for robot
     * 
     * @param joint_state Joint state
     * @param resolution Spatial resolution
     * @param clearance_map Output clearance values
     * @return true if computation successful
     */
    bool computeClearanceMap(
        const sensor_msgs::msg::JointState& joint_state,
        double resolution,
        std::unordered_map<std::string, double>& clearance_map
    );
    
    // ============================================
    // SAFETY AND VALIDATION
    // ============================================
    
    /**
     * @brief Validate safety zones
     * 
     * @param joint_state Joint state to validate
     * @param violated_zones Output list of violated zones
     * @return true if all zones satisfied
     */
    bool validateSafetyZones(
        const sensor_msgs::msg::JointState& joint_state,
        std::vector<SafetyZone>& violated_zones
    );
    
    /**
     * @brief Check if state is in safe region
     * 
     * @param joint_state Joint state to check
     * @param safety_margin Additional safety margin
     * @return true if state is safe
     */
    bool isStateSafe(
        const sensor_msgs::msg::JointState& joint_state,
        double safety_margin = 0.0
    );
    
    /**
     * @brief Check if path is safe
     * 
     * @param trajectory Path to check
     * @param safety_margin Additional safety margin
     * @param violation_times Output times of violations
     * @return true if path is safe
     */
    bool isPathSafe(
        const std::vector<sensor_msgs::msg::JointState>& trajectory,
        double safety_margin,
        std::vector<double>& violation_times
    );
    
    // ============================================
    // VISUALIZATION AND DEBUGGING
    // ============================================
    
    /**
     * @brief Get collision markers for visualization
     * 
     * @param joint_state Joint state
     * @param marker_array Output marker array
     */
    void getCollisionMarkers(
        const sensor_msgs::msg::JointState& joint_state,
        visualization_msgs::msg::MarkerArray& marker_array
    );
    
    /**
     * @brief Get distance field markers
     * 
     * @param bounds Bounding box
     * @param resolution Resolution
     * @param marker_array Output marker array
     */
    void getDistanceFieldMarkers(
        const std::array<Eigen::Vector3d, 2>& bounds,
        double resolution,
        visualization_msgs::msg::MarkerArray& marker_array
    );
    
    /**
     * @brief Get safety zone markers
     * 
     * @param marker_array Output marker array
     */
    void getSafetyZoneMarkers(
        visualization_msgs::msg::MarkerArray& marker_array
    );
    
    // ============================================
    // PERFORMANCE AND MONITORING
    // ============================================
    
    /**
     * @brief Get performance statistics
     * 
     * @param avg_check_time_ms Average check time
     * @param collision_rate Collision detection rate
     * @param cache_hit_rate Cache hit rate
     */
    void getStatistics(
        double& avg_check_time_ms,
        double& collision_rate,
        double& cache_hit_rate
    ) const;
    
    /**
     * @brief Reset statistics
     */
    void resetStatistics();
    
    /**
     * @brief Enable/disable collision checking
     * 
     * @param enabled True to enable
     */
    void setEnabled(bool enabled);
    
    /**
     * @brief Check if collision checker is enabled
     * 
     * @return true if enabled
     */
    bool isEnabled() const;
    
    /**
     * @brief Update configuration at runtime
     * 
     * @param config New configuration
     */
    void updateConfig(const CollisionCheckerConfig& config);
    
private:
    // ============================================
    // PRIVATE TYPES
    // ============================================
    
    struct CacheKey
    {
        std::vector<double> joint_positions;
        std::vector<std::string> active_objects;
        
        bool operator==(const CacheKey& other) const
        {
            return joint_positions == other.joint_positions &&
                   active_objects == other.active_objects;
        }
    };
    
    struct CacheKeyHash
    {
        size_t operator()(const CacheKey& key) const
        {
            size_t hash = 0;
            for (const auto& val : key.joint_positions)
            {
                hash ^= std::hash<double>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            for (const auto& str : key.active_objects)
            {
                hash ^= std::hash<std::string>{}(str) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };
    
    struct CacheEntry
    {
        CollisionCheckResult result;
        rclcpp::Time timestamp;
        size_t access_count;
        
        CacheEntry() : access_count(0) {}
    };
    
    struct CollisionPair
    {
        std::shared_ptr<fcl::CollisionObjectd> obj1;
        std::shared_ptr<fcl::CollisionObjectd> obj2;
        std::string name1;
        std::string name2;
        
        CollisionPair(
            const std::shared_ptr<fcl::CollisionObjectd>& o1,
            const std::shared_ptr<fcl::CollisionObjectd>& o2,
            const std::string& n1,
            const std::string& n2
        ) : obj1(o1), obj2(o2), name1(n1), name2(n2) {}
    };
    
    // ============================================
    // PRIVATE METHODS
    // ============================================
    
    /**
     * @brief Initialize FCL collision manager
     */
    void initializeCollisionManager();
    
    /**
     * @brief Load robot collision geometries from URDF
     * 
     * @param urdf_string URDF description
     * @return true if successful
     */
    bool loadRobotCollisionGeometries(const std::string& urdf_string);
    
    /**
     * @brief Create collision object from geometry
     * 
     * @param name Object name
     * @param geometry FCL geometry
     * @param transform Initial transform
     * @param type Object type
     * @return Shared pointer to collision object
     */
    std::shared_ptr<fcl::CollisionObjectd> createCollisionObject(
        const std::string& name,
        std::shared_ptr<fcl::CollisionGeometryd> geometry,
        const fcl::Transform3d& transform,
        const std::string& type
    );
    
    /**
     * @brief Update robot collision objects for given joint state
     * 
     * @param joint_state Joint state
     */
    void updateRobotCollisionObjects(
        const sensor_msgs::msg::JointState& joint_state
    );
    
    /**
     * @brief Perform actual collision check
     * 
     * @param result Output result
     */
    void performCollisionCheck(CollisionCheckResult& result);
    
    /**
     * @brief Perform continuous collision check
     * 
     * @param start_state Start state
     * @param end_state End state
     * @param result Output result
     */
    void performContinuousCollisionCheck(
        const sensor_msgs::msg::JointState& start_state,
        const sensor_msgs::msg::JointState& end_state,
        ContinuousCollisionCheckResult& result
    );
    
    /**
     * @brief Check cache for existing result
     * 
     * @param key Cache key
     * @param result Output result if found
     * @return true if cache hit
     */
    bool checkCache(const CacheKey& key, CollisionCheckResult& result);
    
    /**
     * @brief Update cache with new result
     * 
     * @param key Cache key
     * @param result Result to cache
     */
    void updateCache(const CacheKey& key, const CollisionCheckResult& result);
    
    /**
     * @brief Clean expired cache entries
     */
    void cleanCache();
    
    /**
     * @brief Worker thread function for parallel checking
     */
    void workerThread();
    
    /**
     * @brief Process collision pair batch
     * 
     * @param pairs Pairs to check
     * @param results Output results
     */
    void processCollisionPairBatch(
        const std::vector<CollisionPair>& pairs,
        std::vector<CollisionCheckResult>& results
    );
    
    /**
     * @brief Check safety zones for given state
     * 
     * @param joint_state Joint state
     * @param violated_zones Output violated zones
     */
    void checkSafetyZones(
        const sensor_msgs::msg::JointState& joint_state,
        std::vector<SafetyZone>& violated_zones
    );
    
    /**
     * @brief Compute forward kinematics for all links
     * 
     * @param joint_state Joint state
     * @param link_transforms Output link transforms
     * @return true if successful
     */
    bool computeLinkTransforms(
        const sensor_msgs::msg::JointState& joint_state,
        std::unordered_map<std::string, fcl::Transform3d>& link_transforms
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
    CollisionCheckerConfig config_;
    std::shared_mutex config_mutex_;
    
    // Robot model
    std::shared_ptr<urdf::Model> urdf_model_;
    std::shared_ptr<srdf::Model> srdf_model_;
    KDL::Chain kinematic_chain_;
    std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
    
    // Collision objects
    std::unordered_map<std::string, std::shared_ptr<fcl::CollisionObjectd>> robot_objects_;
    std::unordered_map<std::string, std::shared_ptr<fcl::CollisionObjectd>> environment_objects_;
    std::unordered_map<std::string, std::shared_ptr<fcl::CollisionObjectd>> dynamic_objects_;
    std::unordered_map<std::string, CollisionObject> object_registry_;
    
    // Collision managers
    std::unique_ptr<fcl::BroadPhaseCollisionManagerd> broadphase_manager_;
    std::unique_ptr<fcl::DynamicAABBTreeCollisionManagerd> dynamic_manager_;
    std::shared_ptr<fcl::CollisionData<double>> collision_data_;
    
    // Safety zones
    std::vector<SafetyZone> safety_zones_;
    std::shared_mutex safety_zones_mutex_;
    
    // Cache
    std::unordered_map<CacheKey, CacheEntry, CacheKeyHash> collision_cache_;
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
    std::vector<double> check_times_;
    size_t total_checks_;
    size_t collision_checks_;
    size_t self_collision_checks_;
    rclcpp::Time last_statistics_publish_;
    
    // State
    std::atomic<bool> enabled_;
    std::atomic<bool> initialized_;
    std::atomic<bool> robot_loaded_;
    
    // ROS2 interfaces
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr performance_pub_;
    
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr obstacle_sub_;
    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
    
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr enable_service_;
    rclcpp::Service<motion_planning_srvs::srv::CheckCollision>::SharedPtr check_service_;
    rclcpp::Service<motion_planning_srvs::srv::GetDistance>::SharedPtr distance_service_;
    
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
     * @brief Convert ROS pose to FCL transform
     * 
     * @param pose ROS pose
     * @return FCL transform
     */
    static fcl::Transform3d poseToFclTransform(const geometry_msgs::msg::Pose& pose);
    
    /**
     * @brief Convert FCL transform to ROS pose
     * 
     * @param transform FCL transform
     * @param pose Output ROS pose
     */
    static void fclTransformToPose(
        const fcl::Transform3d& transform,
        geometry_msgs::msg::Pose& pose
    );
    
    /**
     * @brief Convert KDL frame to FCL transform
     * 
     * @param frame KDL frame
     * @return FCL transform
     */
    static fcl::Transform3d kdlFrameToFclTransform(const KDL::Frame& frame);
    
    /**
     * @brief Create box geometry
     * 
     * @param size Box dimensions
     * @return Shared pointer to geometry
     */
    static std::shared_ptr<fcl::CollisionGeometryd> createBoxGeometry(
        const Eigen::Vector3d& size
    );
    
    /**
     * @brief Create cylinder geometry
     * 
     * @param radius Cylinder radius
     * @param height Cylinder height
     * @return Shared pointer to geometry
     */
    static std::shared_ptr<fcl::CollisionGeometryd> createCylinderGeometry(
        double radius,
        double height
    );
    
    /**
     * @brief Create sphere geometry
     * 
     * @param radius Sphere radius
     * @return Shared pointer to geometry
     */
    static std::shared_ptr<fcl::CollisionGeometryd> createSphereGeometry(double radius);
    
    /**
     * @brief Create mesh geometry from vertices and faces
     * 
     * @param vertices Mesh vertices
     * @param faces Mesh faces
     * @return Shared pointer to geometry
     */
    static std::shared_ptr<fcl::CollisionGeometryd> createMeshGeometry(
        const std::vector<Eigen::Vector3d>& vertices,
        const std::vector<Eigen::Vector3i>& faces
    );
    
    /**
     * @brief Collision callback for FCL
     * 
     * @param o1 First object
     * @param o2 Second object
     * @param data Collision data
     * @return true if collision found
     */
    static bool collisionCallback(
        fcl::CollisionObjectd* o1,
        fcl::CollisionObjectd* o2,
        void* data
    );
    
    /**
     * @brief Distance callback for FCL
     * 
     * @param o1 First object
     * @param o2 Second object
     * @param distance Minimum distance
     * @param p1 Closest point on first object
     * @param p2 Closest point on second object
     * @param data Distance data
     * @return true if distance computed
     */
    static bool distanceCallback(
        fcl::CollisionObjectd* o1,
        fcl::CollisionObjectd* o2,
        double& distance,
        Eigen::Vector3d& p1,
        Eigen::Vector3d& p2,
        void* data
    );
};

}  // namespace motion_planning

// Specialize std::hash for CacheKey
namespace std
{
    template<>
    struct hash<motion_planning::CollisionChecker::CacheKey>
    {
        size_t operator()(const motion_planning::CollisionChecker::CacheKey& key) const
        {
            motion_planning::CollisionChecker::CacheKeyHash hasher;
            return hasher(key);
        }
    };
}

#endif  // MOTION_PLANNING__COLLISION_CHECKER_HPP_
