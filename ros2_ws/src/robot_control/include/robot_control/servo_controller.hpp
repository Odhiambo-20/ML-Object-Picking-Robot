/**
 * @file servo_controller.hpp
 * @brief Production-grade servo motor controller for robotic manipulator
 * 
 * This header defines the ServoController class hierarchy for controlling
 * standard and digital servo motors in industrial pick-and-place robots.
 * 
 * Features:
 * - Multi-servo coordination (up to 16 servos)
 * - PCA9685 16-channel PWM driver support
 * - Direct Raspberry Pi GPIO PWM support
 * - Smooth trajectory generation
 * - Position interpolation and ramping
 * - Collision detection and avoidance
 * - Load configuration from YAML
 * - Thread-safe concurrent control
 * - Comprehensive safety checks
 * - Real-time position feedback
 * - Synchronized multi-joint movements
 * 
 * Hardware Support:
 * - PCA9685 16-Channel 12-bit PWM Driver
 * - Raspberry Pi Hardware PWM (GPIO 12, 13, 18, 19)
 * - Standard analog servos (SG90, MG90S, MG996R, etc.)
 * - Digital servos with position feedback
 * 
 * @author Victor's Production Team
 * @date 2025-01-12
 * @version 1.0.0
 * 
 * @copyright Proprietary - Industrial Use Only
 */

#ifndef ROBOT_CONTROL_SERVO_CONTROLLER_HPP_
#define ROBOT_CONTROL_SERVO_CONTROLLER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <mutex>
#include <thread>
#include <functional>
#include <queue>

#include <cstdint>
#include <cmath>

// Forward declarations for platform-specific interfaces
namespace i2c {
    class I2CDevice;
}

namespace robot_control {

// ============================================================================
// CONSTANTS
// ============================================================================

constexpr int MAX_SERVOS = 16;              ///< Maximum servos per controller
constexpr int PCA9685_ADDRESS = 0x40;       ///< Default PCA9685 I2C address
constexpr int PCA9685_FREQUENCY = 50;       ///< Standard servo PWM frequency (Hz)
constexpr int PWM_RESOLUTION = 4096;        ///< PCA9685 12-bit resolution
constexpr double DEG_TO_RAD = M_PI / 180.0; ///< Degree to radian conversion
constexpr double RAD_TO_DEG = 180.0 / M_PI; ///< Radian to degree conversion

// ============================================================================
// ENUMERATIONS
// ============================================================================

/**
 * @brief Servo controller hardware type
 */
enum class ControllerType {
    PCA9685,        ///< PCA9685 I2C PWM controller
    RPI_GPIO,       ///< Raspberry Pi GPIO PWM
    ARDUINO,        ///< Arduino via serial
    CUSTOM          ///< Custom implementation
};

/**
 * @brief Servo type
 */
enum class ServoType {
    STANDARD,       ///< Standard analog servo (0-180°)
    CONTINUOUS,     ///< Continuous rotation servo
    DIGITAL,        ///< Digital servo with feedback
    LINEAR          ///< Linear servo actuator
};

/**
 * @brief Interpolation method for trajectory generation
 */
enum class InterpolationMethod {
    LINEAR,         ///< Linear interpolation
    CUBIC,          ///< Cubic spline interpolation
    QUINTIC,        ///< Quintic polynomial (smooth accel/decel)
    TRAPEZOIDAL     ///< Trapezoidal velocity profile
};

/**
 * @brief Servo movement state
 */
enum class ServoState {
    IDLE = 0,           ///< Servo idle
    MOVING = 1,         ///< Servo moving to target
    HOLDING = 2,        ///< Servo holding position
    ERROR = 3,          ///< Error condition
    CALIBRATING = 4,    ///< Calibration in progress
    DISABLED = 5        ///< Servo disabled
};

// ============================================================================
// STRUCTURES
// ============================================================================

/**
 * @brief Individual servo configuration
 * Maps to entries in servo_calibration.yaml
 */
struct ServoConfig {
    int id;                     ///< Servo ID/channel (0-15 for PCA9685)
    std::string name;           ///< Servo name (e.g., "base_rotation")
    ServoType type;             ///< Servo type
    
    // Angle limits (degrees)
    double min_angle;           ///< Minimum safe angle
    double max_angle;           ///< Maximum safe angle
    double home_angle;          ///< Home/rest position
    
    // PWM pulse width (microseconds)
    int min_pulse_width;        ///< Pulse width at min angle (typically 500-1000)
    int max_pulse_width;        ///< Pulse width at max angle (typically 2000-2500)
    
    // Calibration
    double offset;              ///< Calibration offset (degrees)
    int direction;              ///< Direction multiplier (1 or -1)
    
    // Limits
    int max_torque;             ///< Max torque percentage (0-100)
    double max_velocity;        ///< Max velocity (degrees/second)
    
    // Physical properties
    double gear_ratio;          ///< Gear ratio
    double mass;                ///< Servo mass (kg) for dynamics
    
    // Runtime state
    std::atomic<double> current_angle;  ///< Current angle
    std::atomic<ServoState> state;      ///< Current state
    std::atomic<bool> enabled;          ///< Is servo enabled
    
    ServoConfig() :
        id(-1), name(""), type(ServoType::STANDARD),
        min_angle(0.0), max_angle(180.0), home_angle(90.0),
        min_pulse_width(500), max_pulse_width(2500),
        offset(0.0), direction(1),
        max_torque(100), max_velocity(60.0),
        gear_ratio(1.0), mass(0.05),
        current_angle(0.0), 
        state(ServoState::IDLE),
        enabled(false)
    {}
};

/**
 * @brief Named position (predefined pose)
 */
struct NamedPosition {
    std::string name;
    std::map<std::string, double> joint_angles;  ///< Joint name -> angle (degrees)
    double transition_time;                       ///< Time to reach position (seconds)
    
    NamedPosition() : transition_time(2.0) {}
};

/**
 * @brief Trajectory point for smooth motion
 */
struct TrajectoryPoint {
    double time;                ///< Time from start (seconds)
    double position;            ///< Target position (degrees)
    double velocity;            ///< Target velocity (degrees/second)
    double acceleration;        ///< Target acceleration (degrees/second²)
    
    TrajectoryPoint() : time(0.0), position(0.0), velocity(0.0), acceleration(0.0) {}
    TrajectoryPoint(double t, double p, double v = 0.0, double a = 0.0) 
        : time(t), position(p), velocity(v), acceleration(a) {}
};

/**
 * @brief Multi-servo trajectory for coordinated motion
 */
struct MultiServoTrajectory {
    std::map<int, std::vector<TrajectoryPoint>> servo_trajectories;  ///< ID -> trajectory
    double total_duration;                                           ///< Total time (seconds)
    InterpolationMethod method;                                      ///< Interpolation method
    std::chrono::steady_clock::time_point start_time;               ///< Start timestamp
    
    MultiServoTrajectory() : total_duration(0.0), method(InterpolationMethod::CUBIC) {}
};

/**
 * @brief Controller configuration
 */
struct ControllerConfig {
    ControllerType type;
    int i2c_bus;                    ///< I2C bus number (typically 1)
    int i2c_address;                ///< I2C address (0x40 for PCA9685)
    int pwm_frequency;              ///< PWM frequency (Hz)
    double update_rate;             ///< Control loop update rate (Hz)
    double position_tolerance;      ///< Position tolerance (degrees)
    bool enable_collision_check;    ///< Enable collision detection
    double safety_margin;           ///< Safety margin from limits (degrees)
    
    ControllerConfig() :
        type(ControllerType::PCA9685),
        i2c_bus(1),
        i2c_address(PCA9685_ADDRESS),
        pwm_frequency(PCA9685_FREQUENCY),
        update_rate(50.0),
        position_tolerance(0.5),
        enable_collision_check(true),
        safety_margin(2.0)
    {}
};

// ============================================================================
// SERVO CONTROLLER BASE CLASS
// ============================================================================

/**
 * @brief Production-grade multi-servo controller
 * 
 * Manages multiple servo motors with:
 * - Smooth trajectory generation
 * - Multi-joint coordination
 * - Safety limit enforcement
 * - Collision detection
 * - Named position presets
 * - Real-time control loop
 */
class ServoController {
public:
    /**
     * @brief Constructor
     * @param config Controller configuration
     */
    explicit ServoController(const ControllerConfig& config = ControllerConfig());
    
    /**
     * @brief Destructor - ensures safe shutdown
     */
    virtual ~ServoController();
    
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    /**
     * @brief Initialize hardware controller
     * @return True if successful
     */
    virtual bool initialize();
    
    /**
     * @brief Shutdown controller safely
     */
    virtual void shutdown();
    
    /**
     * @brief Add servo to controller
     * @param servo_config Servo configuration
     * @return True if added successfully
     */
    bool addServo(const ServoConfig& servo_config);
    
    /**
     * @brief Load configuration from YAML file
     * @param yaml_path Path to servo_calibration.yaml
     * @return True if loaded successfully
     */
    bool loadFromYAML(const std::string& yaml_path);
    
    /**
     * @brief Add named position
     * @param position Named position definition
     */
    void addNamedPosition(const NamedPosition& position);
    
    // ========================================================================
    // SINGLE SERVO CONTROL
    // ========================================================================
    
    /**
     * @brief Set servo angle (blocking)
     * @param servo_id Servo ID
     * @param angle Target angle (degrees)
     * @param duration Transition time (seconds), 0 = immediate
     * @return True if successful
     */
    bool setServoAngle(int servo_id, double angle, double duration = 0.0);
    
    /**
     * @brief Set servo angle by name (blocking)
     * @param servo_name Servo name
     * @param angle Target angle (degrees)
     * @param duration Transition time (seconds)
     * @return True if successful
     */
    bool setServoAngle(const std::string& servo_name, double angle, double duration = 0.0);
    
    /**
     * @brief Set servo velocity
     * @param servo_id Servo ID
     * @param velocity Velocity (degrees/second)
     * @return True if successful
     */
    bool setServoVelocity(int servo_id, double velocity);
    
    /**
     * @brief Get current servo angle
     * @param servo_id Servo ID
     * @return Current angle (degrees)
     */
    double getServoAngle(int servo_id) const;
    
    /**
     * @brief Get current servo angle by name
     * @param servo_name Servo name
     * @return Current angle (degrees)
     */
    double getServoAngle(const std::string& servo_name) const;
    
    /**
     * @brief Disable servo (stops PWM signal)
     * @param servo_id Servo ID
     */
    void disableServo(int servo_id);
    
    /**
     * @brief Enable servo
     * @param servo_id Servo ID
     */
    void enableServo(int servo_id);
    
    // ========================================================================
    // MULTI-SERVO COORDINATED CONTROL
    // ========================================================================
    
    /**
     * @brief Set multiple servos simultaneously (coordinated motion)
     * @param angles Map of servo ID -> target angle (degrees)
     * @param duration Transition time (seconds)
     * @param blocking Wait for completion
     * @return True if successful
     */
    bool setMultipleServos(const std::map<int, double>& angles, 
                           double duration = 2.0,
                           bool blocking = true);
    
    /**
     * @brief Set multiple servos by name
     * @param angles Map of servo name -> target angle
     * @param duration Transition time (seconds)
     * @param blocking Wait for completion
     * @return True if successful
     */
    bool setMultipleServosByName(const std::map<std::string, double>& angles,
                                  double duration = 2.0,
                                  bool blocking = true);
    
    /**
     * @brief Move to named position
     * @param position_name Name of predefined position
     * @param blocking Wait for completion
     * @return True if successful
     */
    bool moveToNamedPosition(const std::string& position_name, bool blocking = true);
    
    /**
     * @brief Home all servos
     * @param blocking Wait for completion
     * @return True if successful
     */
    bool homeAll(bool blocking = true);
    
    /**
     * @brief Stop all servo motion
     */
    void stopAll();
    
    /**
     * @brief Emergency stop - halt all motion immediately
     */
    void emergencyStop();
    
    // ========================================================================
    // TRAJECTORY CONTROL
    // ========================================================================
    
    /**
     * @brief Execute multi-servo trajectory
     * @param trajectory Pre-computed trajectory
     * @param blocking Wait for completion
     * @return True if successful
     */
    bool executeTrajectory(const MultiServoTrajectory& trajectory, bool blocking = true);
    
    /**
     * @brief Generate smooth trajectory between positions
     * @param start_angles Starting angles
     * @param end_angles Target angles
     * @param duration Total duration (seconds)
     * @param method Interpolation method
     * @return Generated trajectory
     */
    MultiServoTrajectory generateTrajectory(
        const std::map<int, double>& start_angles,
        const std::map<int, double>& end_angles,
        double duration,
        InterpolationMethod method = InterpolationMethod::CUBIC
    );
    
    // ========================================================================
    // STATE QUERIES
    // ========================================================================
    
    /**
     * @brief Check if servo is moving
     * @param servo_id Servo ID
     * @return True if moving
     */
    bool isServoMoving(int servo_id) const;
    
    /**
     * @brief Check if any servo is moving
     * @return True if any servo moving
     */
    bool isAnyServoMoving() const;
    
    /**
     * @brief Get servo state
     * @param servo_id Servo ID
     * @return Servo state
     */
    ServoState getServoState(int servo_id) const;
    
    /**
     * @brief Get all servo angles
     * @return Map of servo ID -> current angle
     */
    std::map<int, double> getAllServoAngles() const;
    
    /**
     * @brief Check if controller is ready
     * @return True if ready for commands
     */
    bool isReady() const { return is_initialized_ && !emergency_stop_active_; }
    
    /**
     * @brief Get named position list
     * @return Vector of position names
     */
    std::vector<std::string> getNamedPositions() const;
    
    // ========================================================================
    // SAFETY AND VALIDATION
    // ========================================================================
    
    /**
     * @brief Check if angle is within safe limits
     * @param servo_id Servo ID
     * @param angle Angle to check (degrees)
     * @return True if safe
     */
    bool isAngleSafe(int servo_id, double angle) const;
    
    /**
     * @brief Enable/disable safety checks
     * @param enable True to enable
     */
    void enableSafetyChecks(bool enable) { safety_checks_enabled_ = enable; }
    
    /**
     * @brief Set collision check callback
     * @param callback Function to check for collisions
     */
    void setCollisionCheckCallback(
        std::function<bool(const std::map<int, double>&)> callback
    ) {
        collision_check_callback_ = callback;
    }
    
    // ========================================================================
    // CALIBRATION
    // ========================================================================
    
    /**
     * @brief Calibrate single servo (find center, limits)
     * @param servo_id Servo ID
     * @return True if successful
     */
    bool calibrateServo(int servo_id);
    
    /**
     * @brief Calibrate all servos
     * @return True if all successful
     */
    bool calibrateAll();
    
    /**
     * @brief Save current positions as calibration
     * @param filename Output YAML file
     * @return True if successful
     */
    bool saveCalibration(const std::string& filename);

protected:
    // ========================================================================
    // PROTECTED METHODS
    // ========================================================================
    
    /**
     * @brief Control loop (runs at update_rate)
     */
    void controlLoop();
    
    /**
     * @brief Update servo position
     * @param servo_id Servo ID
     * @param angle Target angle (degrees)
     */
    virtual void updateServoPosition(int servo_id, double angle);
    
    /**
     * @brief Convert angle to PWM pulse width
     * @param servo_id Servo ID
     * @param angle Angle (degrees)
     * @return Pulse width (microseconds)
     */
    int angleToPulseWidth(int servo_id, double angle) const;
    
    /**
     * @brief Convert pulse width to angle
     * @param servo_id Servo ID
     * @param pulse_width Pulse width (microseconds)
     * @return Angle (degrees)
     */
    double pulseWidthToAngle(int servo_id, int pulse_width) const;
    
    /**
     * @brief Write PWM to hardware
     * @param channel PWM channel
     * @param pulse_width Pulse width (microseconds)
     */
    virtual void writePWM(int channel, int pulse_width) = 0;
    
    /**
     * @brief Interpolate trajectory point
     * @param trajectory Trajectory to interpolate
     * @param servo_id Servo ID
     * @param elapsed_time Time since trajectory start (seconds)
     * @return Interpolated angle
     */
    double interpolateTrajectory(
        const MultiServoTrajectory& trajectory,
        int servo_id,
        double elapsed_time
    ) const;
    
    /**
     * @brief Check trajectory for safety
     * @param trajectory Trajectory to validate
     * @return True if safe
     */
    bool validateTrajectory(const MultiServoTrajectory& trajectory) const;
    
    /**
     * @brief Constrain angle to limits
     * @param servo_id Servo ID
     * @param angle Angle to constrain
     * @return Constrained angle
     */
    double constrainAngle(int servo_id, double angle) const;
    
    // ========================================================================
    // PROTECTED MEMBER VARIABLES
    // ========================================================================
    
    ControllerConfig config_;
    
    // Servo management
    std::map<int, ServoConfig> servos_;                     ///< ID -> config
    std::unordered_map<std::string, int> servo_name_to_id_; ///< Name -> ID
    std::map<std::string, NamedPosition> named_positions_;  ///< Name -> position
    
    // State
    std::atomic<bool> is_initialized_;
    std::atomic<bool> is_running_;
    std::atomic<bool> emergency_stop_active_;
    std::atomic<bool> safety_checks_enabled_;
    
    // Trajectory execution
    std::unique_ptr<MultiServoTrajectory> active_trajectory_;
    std::chrono::steady_clock::time_point trajectory_start_time_;
    
    // Thread management
    std::unique_ptr<std::thread> control_thread_;
    mutable std::mutex servo_mutex_;
    mutable std::mutex trajectory_mutex_;
    
    // Callbacks
    std::function<bool(const std::map<int, double>&)> collision_check_callback_;
    
    // Hardware interface
    std::shared_ptr<void> hardware_interface_;  ///< Platform-specific
    
    // Statistics
    size_t position_updates_;
    size_t trajectory_executions_;
    std::chrono::steady_clock::time_point start_time_;
};

// ============================================================================
// PCA9685 SERVO CONTROLLER IMPLEMENTATION
// ============================================================================

/**
 * @brief Concrete implementation for PCA9685 16-channel PWM driver
 */
class PCA9685ServoController : public ServoController {
public:
    /**
     * @brief Constructor
     * @param config Controller configuration
     */
    explicit PCA9685ServoController(const ControllerConfig& config = ControllerConfig());
    
    /**
     * @brief Destructor
     */
    ~PCA9685ServoController() override;
    
    /**
     * @brief Initialize PCA9685 hardware
     * @return True if successful
     */
    bool initialize() override;
    
    /**
     * @brief Shutdown PCA9685
     */
    void shutdown() override;

protected:
    /**
     * @brief Write PWM to PCA9685 channel
     * @param channel Channel (0-15)
     * @param pulse_width Pulse width (microseconds)
     */
    void writePWM(int channel, int pulse_width) override;

private:
    /**
     * @brief Set PCA9685 PWM frequency
     * @param frequency Frequency (Hz)
     */
    void setPWMFrequency(int frequency);
    
    /**
     * @brief Write to PCA9685 register
     * @param reg Register address
     * @param value Value to write
     */
    void writeRegister(uint8_t reg, uint8_t value);
    
    /**
     * @brief Read from PCA9685 register
     * @param reg Register address
     * @return Register value
     */
    uint8_t readRegister(uint8_t reg);
    
    std::shared_ptr<i2c::I2CDevice> i2c_device_;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Convert servo state to string
 */
inline std::string toString(ServoState state) {
    switch (state) {
        case ServoState::IDLE: return "IDLE";
        case ServoState::MOVING: return "MOVING";
        case ServoState::HOLDING: return "HOLDING";
        case ServoState::ERROR: return "ERROR";
        case ServoState::CALIBRATING: return "CALIBRATING";
        case ServoState::DISABLED: return "DISABLED";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Smooth step function (S-curve)
 * @param t Time parameter (0.0 - 1.0)
 * @return Smoothed value (0.0 - 1.0)
 */
inline double smoothStep(double t) {
    t = std::max(0.0, std::min(1.0, t));
    return t * t * (3.0 - 2.0 * t);
}

/**
 * @brief Smoother step function (quintic)
 * @param t Time parameter (0.0 - 1.0)
 * @return Smoothed value (0.0 - 1.0)
 */
inline double smootherStep(double t) {
    t = std::max(0.0, std::min(1.0, t));
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

} // namespace robot_control

#endif // ROBOT_CONTROL_SERVO_CONTROLLER_HPP_
