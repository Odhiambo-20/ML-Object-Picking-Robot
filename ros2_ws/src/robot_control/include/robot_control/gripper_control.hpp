/**
 * @file gripper_control.hpp
 * @brief Production-grade gripper control for industrial pick-and-place robot
 * 
 * This header defines the GripperController class for managing the robot's
 * end effector with advanced features including:
 * - Precise position control
 * - Force sensing and feedback
 * - Object slip detection
 * - Adaptive grasping strategies
 * - Safety monitoring
 * 
 * @author Victor's Production Team
 * @date 2025-01-12
 * @version 1.0.0
 * 
 * @copyright Proprietary - Industrial Use Only
 */

#ifndef ROBOT_CONTROL_GRIPPER_CONTROL_HPP_
#define ROBOT_CONTROL_GRIPPER_CONTROL_HPP_

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <functional>
#include <mutex>
#include <atomic>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/bool.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "robot_interfaces/msg/gripper_state.hpp"
#include "robot_interfaces/srv/grasp_object.hpp"

namespace robot_control {

/**
 * @brief Gripper operating states
 */
enum class GripperState {
    IDLE = 0,           ///< Gripper is idle, ready for commands
    OPENING = 1,        ///< Gripper is opening
    CLOSING = 2,        ///< Gripper is closing
    HOLDING = 3,        ///< Gripper is holding an object
    GRIPPING = 4,       ///< Gripper is actively gripping
    ERROR = 5,          ///< Gripper encountered an error
    CALIBRATING = 6,    ///< Gripper is calibrating
    EMERGENCY_STOP = 7  ///< Emergency stop activated
};

/**
 * @brief Grasp strategies for different object types
 */
enum class GraspStrategy {
    PARALLEL_JAW = 0,   ///< Standard parallel jaw grasp
    PINCH = 1,          ///< Pinch grasp for small objects
    POWER = 2,          ///< Power grasp with maximum force
    PRECISION = 3,      ///< Precision grasp with minimal force
    ADAPTIVE = 4,       ///< Adaptive grasp based on object properties
    ENVELOPE = 5        ///< Envelope grasp for irregular shapes
};

/**
 * @brief Object properties for adaptive grasping
 */
struct ObjectProperties {
    double estimated_mass;      ///< Estimated mass (kg)
    double friction_coefficient; ///< Surface friction coefficient
    std::string material_type;  ///< Material (e.g., "metal", "plastic")
    bool is_fragile;            ///< Whether object is fragile
    double width;               ///< Object width (meters)
    double height;              ///< Object height (meters)
    bool is_slippery;           ///< Whether object is slippery
};

/**
 * @brief Gripper configuration parameters
 */
struct GripperConfig {
    // Physical parameters
    double max_opening_width;      ///< Maximum gripper opening (meters)
    double min_opening_width;      ///< Minimum gripper opening (meters)
    double max_force;              ///< Maximum gripping force (N)
    double min_force;              ///< Minimum gripping force (N)
    
    // Control parameters
    double position_tolerance;     ///< Position control tolerance (meters)
    double force_tolerance;        ///< Force control tolerance (N)
    double closing_velocity;       ///< Closing velocity (m/s)
    double opening_velocity;       ///< Opening velocity (m/s)
    
    // Safety parameters
    double max_current;            ///< Maximum motor current (A)
    double stall_current_threshold; ///< Current threshold for stall detection (A)
    double slip_detection_threshold; ///< Threshold for slip detection
    double timeout_duration;       ///< Command timeout (seconds)
    
    // Sensor parameters
    bool has_force_sensor;         ///< Force sensor available
    bool has_position_encoder;     ///< Position encoder available
    bool has_current_sensor;       ///< Current sensor available
    bool has_tactile_sensors;      ///< Tactile sensors available
    
    // Default values
    GripperConfig() :
        max_opening_width(0.085),      // 85mm
        min_opening_width(0.0),
        max_force(50.0),               // 50N
        min_force(1.0),                // 1N
        position_tolerance(0.001),     // 1mm
        force_tolerance(0.5),          // 0.5N
        closing_velocity(0.05),        // 50mm/s
        opening_velocity(0.08),        // 80mm/s
        max_current(2.0),              // 2A
        stall_current_threshold(1.5),  // 1.5A
        slip_detection_threshold(0.15),
        timeout_duration(5.0),
        has_force_sensor(false),
        has_position_encoder(true),
        has_current_sensor(true),
        has_tactile_sensors(false)
    {}
};

/**
 * @brief Production-grade gripper controller class
 * 
 * This class provides comprehensive gripper control with:
 * - Position and force control modes
 * - Adaptive grasping strategies
 * - Object slip detection
 * - Safety monitoring
 * - ROS2 integration
 */
class GripperController : public rclcpp::Node {
public:
    /**
     * @brief Constructor
     * @param options Node options for ROS2
     */
    explicit GripperController(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    
    /**
     * @brief Destructor - ensures safe shutdown
     */
    virtual ~GripperController();
    
    // ========================================================================
    // PUBLIC CONTROL METHODS
    // ========================================================================
    
    /**
     * @brief Open gripper to specified width
     * @param width Target opening width (meters)
     * @param blocking If true, wait for completion
     * @return True if command accepted
     */
    bool open(double width = -1.0, bool blocking = false);
    
    /**
     * @brief Close gripper with specified force
     * @param force Target gripping force (N), -1 for automatic
     * @param blocking If true, wait for completion
     * @return True if command accepted
     */
    bool close(double force = -1.0, bool blocking = false);
    
    /**
     * @brief Grasp object using adaptive strategy
     * @param properties Object properties for adaptive control
     * @param strategy Grasping strategy to use
     * @return True if grasp successful
     */
    bool grasp(const ObjectProperties& properties, 
               GraspStrategy strategy = GraspStrategy::ADAPTIVE);
    
    /**
     * @brief Release object safely
     * @param open_fully If true, open to maximum width
     * @return True if release successful
     */
    bool release(bool open_fully = true);
    
    /**
     * @brief Stop gripper motion immediately
     */
    void stop();
    
    /**
     * @brief Emergency stop - halt all motion and enter safe state
     */
    void emergencyStop();
    
    /**
     * @brief Calibrate gripper (find limits, zero position)
     * @return True if calibration successful
     */
    bool calibrate();
    
    /**
     * @brief Home gripper to default position
     * @return True if homing successful
     */
    bool home();
    
    // ========================================================================
    // STATE QUERY METHODS
    // ========================================================================
    
    /**
     * @brief Get current gripper state
     * @return Current state
     */
    GripperState getState() const { return current_state_; }
    
    /**
     * @brief Check if gripper is currently holding an object
     * @return True if object is grasped
     */
    bool isHolding() const { return is_holding_object_; }
    
    /**
     * @brief Get current gripper opening width
     * @return Opening width in meters
     */
    double getCurrentWidth() const { return current_width_; }
    
    /**
     * @brief Get current gripping force
     * @return Force in Newtons
     */
    double getCurrentForce() const { return current_force_; }
    
    /**
     * @brief Get motor current
     * @return Current in Amperes
     */
    double getMotorCurrent() const { return motor_current_; }
    
    /**
     * @brief Check if object slip is detected
     * @return True if slip detected
     */
    bool isSlipping() const { return slip_detected_; }
    
    /**
     * @brief Check if gripper is ready for new command
     * @return True if ready
     */
    bool isReady() const { 
        return current_state_ == GripperState::IDLE || 
               current_state_ == GripperState::HOLDING; 
    }
    
    // ========================================================================
    // CONFIGURATION METHODS
    // ========================================================================
    
    /**
     * @brief Set gripper configuration
     * @param config New configuration
     */
    void setConfiguration(const GripperConfig& config);
    
    /**
     * @brief Get current configuration
     * @return Current configuration
     */
    GripperConfig getConfiguration() const { return config_; }
    
    /**
     * @brief Set force limit for safety
     * @param max_force Maximum force (N)
     */
    void setMaxForce(double max_force);
    
    /**
     * @brief Enable/disable slip detection
     * @param enable True to enable
     */
    void enableSlipDetection(bool enable) { slip_detection_enabled_ = enable; }
    
    /**
     * @brief Set callback for object detected event
     * @param callback Callback function
     */
    void setObjectDetectedCallback(std::function<void(bool)> callback) {
        object_detected_callback_ = callback;
    }
    
    /**
     * @brief Set callback for slip detected event
     * @param callback Callback function
     */
    void setSlipDetectedCallback(std::function<void()> callback) {
        slip_detected_callback_ = callback;
    }

private:
    // ========================================================================
    // PRIVATE METHODS
    // ========================================================================
    
    /**
     * @brief Initialize ROS2 publishers, subscribers, services
     */
    void initializeROS();
    
    /**
     * @brief Initialize hardware interface
     * @return True if successful
     */
    bool initializeHardware();
    
    /**
     * @brief Load parameters from ROS2 parameter server
     */
    void loadParameters();
    
    /**
     * @brief Main control loop (runs at fixed rate)
     */
    void controlLoop();
    
    /**
     * @brief Update gripper state based on sensor feedback
     */
    void updateState();
    
    /**
     * @brief Monitor for object slip
     */
    void monitorSlip();
    
    /**
     * @brief Detect object contact
     * @return True if object detected
     */
    bool detectObject();
    
    /**
     * @brief Calculate required force for object
     * @param properties Object properties
     * @return Required force (N)
     */
    double calculateRequiredForce(const ObjectProperties& properties);
    
    /**
     * @brief Execute position control
     * @param target_width Target width (meters)
     * @return True if reached
     */
    bool executePositionControl(double target_width);
    
    /**
     * @brief Execute force control
     * @param target_force Target force (N)
     * @return True if reached
     */
    bool executeForceControl(double target_force);
    
    /**
     * @brief Publish current gripper state
     */
    void publishState();
    
    /**
     * @brief Check safety conditions
     * @return True if safe to continue
     */
    bool checkSafety();
    
    /**
     * @brief Handle hardware errors
     * @param error_msg Error message
     */
    void handleError(const std::string& error_msg);
    
    // ========================================================================
    // ROS2 CALLBACKS
    // ========================================================================
    
    /**
     * @brief Position command callback
     */
    void positionCommandCallback(const std_msgs::msg::Float64::SharedPtr msg);
    
    /**
     * @brief Force command callback
     */
    void forceCommandCallback(const std_msgs::msg::Float64::SharedPtr msg);
    
    /**
     * @brief Grasp service callback
     */
    void graspServiceCallback(
        const std::shared_ptr<robot_interfaces::srv::GraspObject::Request> request,
        std::shared_ptr<robot_interfaces::srv::GraspObject::Response> response);
    
    /**
     * @brief Emergency stop callback
     */
    void emergencyStopCallback(const std_msgs::msg::Bool::SharedPtr msg);
    
    // ========================================================================
    // MEMBER VARIABLES
    // ========================================================================
    
    // Configuration
    GripperConfig config_;
    
    // State variables
    std::atomic<GripperState> current_state_;
    std::atomic<double> current_width_;
    std::atomic<double> current_force_;
    std::atomic<double> motor_current_;
    std::atomic<bool> is_holding_object_;
    std::atomic<bool> slip_detected_;
    std::atomic<bool> slip_detection_enabled_;
    
    // Target values
    double target_width_;
    double target_force_;
    GraspStrategy current_strategy_;
    
    // Timing
    std::chrono::steady_clock::time_point last_command_time_;
    std::chrono::steady_clock::time_point grasp_start_time_;
    
    // Thread safety
    mutable std::mutex state_mutex_;
    mutable std::mutex command_mutex_;
    
    // ROS2 components
    rclcpp::TimerBase::SharedPtr control_timer_;
    rclcpp::Publisher<robot_interfaces::msg::GripperState>::SharedPtr state_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr position_cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr force_cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr emergency_stop_sub_;
    rclcpp::Service<robot_interfaces::srv::GraspObject>::SharedPtr grasp_service_;
    
    // Callbacks
    std::function<void(bool)> object_detected_callback_;
    std::function<void()> slip_detected_callback_;
    
    // Hardware interface pointer (implementation-specific)
    std::shared_ptr<void> hardware_interface_;
    
    // Control loop frequency (Hz)
    static constexpr double CONTROL_FREQUENCY = 50.0;
    
    // Slip detection variables
    std::vector<double> force_history_;
    size_t force_history_size_;
    double last_stable_force_;
    
    // Safety flags
    std::atomic<bool> emergency_stop_active_;
    std::atomic<bool> calibration_complete_;
    
    // Statistics and monitoring
    size_t successful_grasps_;
    size_t failed_grasps_;
    size_t slip_events_;
    std::chrono::steady_clock::time_point node_start_time_;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Convert gripper state to string
 * @param state Gripper state
 * @return String representation
 */
inline std::string toString(GripperState state) {
    switch (state) {
        case GripperState::IDLE: return "IDLE";
        case GripperState::OPENING: return "OPENING";
        case GripperState::CLOSING: return "CLOSING";
        case GripperState::HOLDING: return "HOLDING";
        case GripperState::GRIPPING: return "GRIPPING";
        case GripperState::ERROR: return "ERROR";
        case GripperState::CALIBRATING: return "CALIBRATING";
        case GripperState::EMERGENCY_STOP: return "EMERGENCY_STOP";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert grasp strategy to string
 * @param strategy Grasp strategy
 * @return String representation
 */
inline std::string toString(GraspStrategy strategy) {
    switch (strategy) {
        case GraspStrategy::PARALLEL_JAW: return "PARALLEL_JAW";
        case GraspStrategy::PINCH: return "PINCH";
        case GraspStrategy::POWER: return "POWER";
        case GraspStrategy::PRECISION: return "PRECISION";
        case GraspStrategy::ADAPTIVE: return "ADAPTIVE";
        case GraspStrategy::ENVELOPE: return "ENVELOPE";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Calculate grip force based on object mass and friction
 * @param mass Object mass (kg)
 * @param friction Friction coefficient
 * @param safety_factor Safety multiplier
 * @return Required force (N)
 */
inline double calculateGripForce(double mass, double friction, 
                                 double safety_factor = 2.0) {
    const double g = 9.81; // Gravity (m/s^2)
    return (mass * g * safety_factor) / friction;
}

} // namespace robot_control

#endif // ROBOT_CONTROL_GRIPPER_CONTROL_HPP_
