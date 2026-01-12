/**
 * @file motor_driver.hpp
 * @brief Production-grade motor driver interface for L298N and compatible drivers
 * 
 * This header defines the MotorDriver class hierarchy for controlling DC motors
 * in industrial robotics applications. Supports:
 * - L298N H-Bridge motor driver
 * - PWM speed control
 * - Direction control
 * - Current sensing and protection
 * - Encoder feedback integration
 * - PID velocity control
 * - Safety and fault detection
 * 
 * Hardware Support:
 * - L298N Dual H-Bridge Driver
 * - TB6612FNG Motor Driver
 * - DRV8833 Motor Driver
 * - Custom H-Bridge implementations
 * 
 * @author Victor's Production Team
 * @date 2025-01-12
 * @version 1.0.0
 * 
 * @copyright Proprietary - Industrial Use Only
 */

#ifndef ROBOT_CONTROL_MOTOR_DRIVER_HPP_
#define ROBOT_CONTROL_MOTOR_DRIVER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <mutex>
#include <functional>

#include <cstdint>
#include <cmath>

namespace robot_control {

// ============================================================================
// ENUMERATIONS
// ============================================================================

/**
 * @brief Motor rotation direction
 */
enum class MotorDirection {
    FORWARD = 1,    ///< Clockwise rotation
    BACKWARD = -1,  ///< Counter-clockwise rotation
    STOPPED = 0,    ///< Motor stopped
    BRAKE = 2       ///< Active braking
};

/**
 * @brief Motor driver state
 */
enum class MotorState {
    IDLE = 0,           ///< Motor idle, ready for commands
    RUNNING = 1,        ///< Motor running normally
    BRAKING = 2,        ///< Motor actively braking
    FAULT = 3,          ///< Driver fault detected
    OVERCURRENT = 4,    ///< Overcurrent condition
    OVERTEMPERATURE = 5,///< Overtemperature condition
    STALLED = 6,        ///< Motor stalled (blocked)
    DISABLED = 7        ///< Driver disabled
};

/**
 * @brief Control mode for motor
 */
enum class ControlMode {
    OPEN_LOOP = 0,      ///< Open-loop speed control (no feedback)
    CLOSED_LOOP = 1,    ///< Closed-loop with encoder feedback
    POSITION = 2,       ///< Position control mode
    TORQUE = 3          ///< Torque/current control mode
};

/**
 * @brief Motor driver type
 */
enum class DriverType {
    L298N,              ///< L298N Dual H-Bridge
    TB6612FNG,          ///< TB6612FNG Motor Driver
    DRV8833,            ///< DRV8833 Motor Driver
    CUSTOM              ///< Custom implementation
};

// ============================================================================
// STRUCTURES
// ============================================================================

/**
 * @brief Motor hardware configuration
 */
struct MotorConfig {
    // GPIO pins
    int pwm_pin;                ///< PWM control pin
    int dir1_pin;               ///< Direction control pin 1 (IN1/AIN1)
    int dir2_pin;               ///< Direction control pin 2 (IN2/AIN2)
    int enable_pin;             ///< Enable pin (optional, -1 if not used)
    int current_sense_pin;      ///< Current sense analog pin (optional)
    int encoder_a_pin;          ///< Encoder A channel (optional)
    int encoder_b_pin;          ///< Encoder B channel (optional)
    
    // Motor specifications
    double rated_voltage;       ///< Rated voltage (V)
    double rated_current;       ///< Rated current (A)
    double max_current;         ///< Maximum safe current (A)
    double stall_current;       ///< Stall current threshold (A)
    double max_rpm;             ///< Maximum RPM
    double gear_ratio;          ///< Gear reduction ratio
    int encoder_ppr;            ///< Encoder pulses per revolution
    
    // PWM parameters
    int pwm_frequency;          ///< PWM frequency (Hz)
    int pwm_resolution;         ///< PWM resolution (bits)
    double pwm_dead_zone;       ///< Dead zone (0.0 - 1.0)
    
    // Safety limits
    double max_acceleration;    ///< Max acceleration (RPM/s)
    double max_deceleration;    ///< Max deceleration (RPM/s)
    double emergency_stop_time; ///< Emergency stop time (s)
    
    // Control parameters
    double current_sense_gain;  ///< Current sensor gain (V/A)
    double current_sense_offset;///< Current sensor offset (V)
    bool invert_direction;      ///< Invert direction control
    bool invert_pwm;            ///< Invert PWM signal
    
    DriverType driver_type;     ///< Driver IC type
    
    // Default constructor
    MotorConfig() :
        pwm_pin(-1), dir1_pin(-1), dir2_pin(-1), 
        enable_pin(-1), current_sense_pin(-1),
        encoder_a_pin(-1), encoder_b_pin(-1),
        rated_voltage(6.0), rated_current(1.0),
        max_current(2.0), stall_current(1.5),
        max_rpm(200.0), gear_ratio(1.0), encoder_ppr(0),
        pwm_frequency(1000), pwm_resolution(8),
        pwm_dead_zone(0.05), max_acceleration(500.0),
        max_deceleration(1000.0), emergency_stop_time(0.5),
        current_sense_gain(0.5), current_sense_offset(0.0),
        invert_direction(false), invert_pwm(false),
        driver_type(DriverType::L298N)
    {}
};

/**
 * @brief PID controller parameters
 */
struct PIDParameters {
    double kp;          ///< Proportional gain
    double ki;          ///< Integral gain
    double kd;          ///< Derivative gain
    double max_output;  ///< Maximum output (0.0 - 1.0)
    double min_output;  ///< Minimum output
    double max_integral;///< Integral windup limit
    
    PIDParameters() : 
        kp(1.0), ki(0.1), kd(0.05),
        max_output(1.0), min_output(-1.0),
        max_integral(10.0)
    {}
};

/**
 * @brief Motor telemetry data
 */
struct MotorTelemetry {
    double velocity;            ///< Current velocity (RPM)
    double position;            ///< Current position (radians or revolutions)
    double current;             ///< Motor current (A)
    double voltage;             ///< Applied voltage (V)
    double duty_cycle;          ///< PWM duty cycle (0.0 - 1.0)
    double temperature;         ///< Motor temperature (°C, if available)
    MotorDirection direction;   ///< Current direction
    MotorState state;           ///< Current state
    int64_t encoder_count;      ///< Raw encoder count
    
    std::chrono::steady_clock::time_point timestamp; ///< Data timestamp
    
    MotorTelemetry() :
        velocity(0.0), position(0.0), current(0.0),
        voltage(0.0), duty_cycle(0.0), temperature(0.0),
        direction(MotorDirection::STOPPED),
        state(MotorState::IDLE), encoder_count(0),
        timestamp(std::chrono::steady_clock::now())
    {}
};

// ============================================================================
// MOTOR DRIVER BASE CLASS
// ============================================================================

/**
 * @brief Abstract base class for motor drivers
 * 
 * Provides common interface for all motor driver implementations.
 * Production-grade features include:
 * - Thread-safe operation
 * - Real-time safety monitoring
 * - Current sensing and protection
 * - Encoder feedback integration
 * - PID velocity control
 * - Comprehensive error handling
 */
class MotorDriver {
public:
    /**
     * @brief Constructor
     * @param config Motor configuration
     * @param name Motor name for logging
     */
    explicit MotorDriver(const MotorConfig& config, const std::string& name = "motor");
    
    /**
     * @brief Virtual destructor
     */
    virtual ~MotorDriver();
    
    // ========================================================================
    // INITIALIZATION AND SHUTDOWN
    // ========================================================================
    
    /**
     * @brief Initialize motor driver hardware
     * @return True if successful
     */
    virtual bool initialize() = 0;
    
    /**
     * @brief Shutdown motor driver safely
     */
    virtual void shutdown();
    
    /**
     * @brief Reset driver to initial state
     */
    virtual void reset();
    
    /**
     * @brief Enable motor driver
     * @return True if successful
     */
    virtual bool enable();
    
    /**
     * @brief Disable motor driver
     */
    virtual void disable();
    
    // ========================================================================
    // MOTION CONTROL
    // ========================================================================
    
    /**
     * @brief Set motor speed (open-loop)
     * @param speed Speed (-1.0 to 1.0, normalized)
     * @return True if command accepted
     */
    virtual bool setSpeed(double speed);
    
    /**
     * @brief Set motor velocity with closed-loop control
     * @param velocity Target velocity (RPM)
     * @return True if command accepted
     */
    virtual bool setVelocity(double velocity);
    
    /**
     * @brief Set motor direction
     * @param direction Target direction
     */
    virtual void setDirection(MotorDirection direction);
    
    /**
     * @brief Stop motor (coast to stop)
     */
    virtual void stop();
    
    /**
     * @brief Brake motor (active braking)
     */
    virtual void brake();
    
    /**
     * @brief Emergency stop (maximum deceleration)
     */
    virtual void emergencyStop();
    
    // ========================================================================
    // STATE AND TELEMETRY
    // ========================================================================
    
    /**
     * @brief Get current motor state
     * @return Motor state
     */
    MotorState getState() const { return current_state_; }
    
    /**
     * @brief Get complete motor telemetry
     * @return Telemetry structure
     */
    MotorTelemetry getTelemetry() const;
    
    /**
     * @brief Get current velocity
     * @return Velocity in RPM
     */
    double getVelocity() const { return current_velocity_; }
    
    /**
     * @brief Get current position
     * @return Position in radians
     */
    double getPosition() const { return current_position_; }
    
    /**
     * @brief Get motor current
     * @return Current in Amperes
     */
    double getCurrent() const { return current_current_; }
    
    /**
     * @brief Get PWM duty cycle
     * @return Duty cycle (0.0 - 1.0)
     */
    double getDutyCycle() const { return current_duty_cycle_; }
    
    /**
     * @brief Check if motor is enabled
     * @return True if enabled
     */
    bool isEnabled() const { return is_enabled_; }
    
    /**
     * @brief Check if motor is in fault state
     * @return True if fault detected
     */
    bool hasFault() const { 
        return current_state_ == MotorState::FAULT ||
               current_state_ == MotorState::OVERCURRENT ||
               current_state_ == MotorState::OVERTEMPERATURE;
    }
    
    // ========================================================================
    // CONFIGURATION
    // ========================================================================
    
    /**
     * @brief Set PID parameters for closed-loop control
     * @param params PID parameters
     */
    void setPIDParameters(const PIDParameters& params);
    
    /**
     * @brief Get current PID parameters
     * @return PID parameters
     */
    PIDParameters getPIDParameters() const { return pid_params_; }
    
    /**
     * @brief Set control mode
     * @param mode Control mode
     */
    void setControlMode(ControlMode mode) { control_mode_ = mode; }
    
    /**
     * @brief Get control mode
     * @return Current control mode
     */
    ControlMode getControlMode() const { return control_mode_; }
    
    /**
     * @brief Set current limit
     * @param limit Current limit (A)
     */
    void setCurrentLimit(double limit);
    
    /**
     * @brief Set velocity ramp rate
     * @param acceleration Acceleration rate (RPM/s)
     * @param deceleration Deceleration rate (RPM/s)
     */
    void setRampRate(double acceleration, double deceleration);
    
    // ========================================================================
    // CALLBACKS
    // ========================================================================
    
    /**
     * @brief Set fault callback
     * @param callback Function called on fault
     */
    void setFaultCallback(std::function<void(MotorState)> callback) {
        fault_callback_ = callback;
    }
    
    /**
     * @brief Set overcurrent callback
     * @param callback Function called on overcurrent
     */
    void setOvercurrentCallback(std::function<void(double)> callback) {
        overcurrent_callback_ = callback;
    }
    
    // ========================================================================
    // ENCODER METHODS (if encoder available)
    // ========================================================================
    
    /**
     * @brief Reset encoder count
     */
    virtual void resetEncoder();
    
    /**
     * @brief Get raw encoder count
     * @return Encoder pulses
     */
    int64_t getEncoderCount() const { return encoder_count_; }
    
    /**
     * @brief Check if encoder is available
     * @return True if encoder configured
     */
    bool hasEncoder() const { 
        return config_.encoder_a_pin >= 0 && config_.encoder_b_pin >= 0; 
    }

protected:
    // ========================================================================
    // PROTECTED METHODS (for derived classes)
    // ========================================================================
    
    /**
     * @brief Update motor state (called by control loop)
     */
    virtual void updateState();
    
    /**
     * @brief Read current sensor
     * @return Current in Amperes
     */
    virtual double readCurrent();
    
    /**
     * @brief Read encoder
     * @return Velocity in RPM
     */
    virtual double readEncoder();
    
    /**
     * @brief Write PWM value to hardware
     * @param duty_cycle Duty cycle (0.0 - 1.0)
     */
    virtual void writePWM(double duty_cycle) = 0;
    
    /**
     * @brief Set hardware direction pins
     * @param direction Direction to set
     */
    virtual void setHardwareDirection(MotorDirection direction) = 0;
    
    /**
     * @brief Compute PID control output
     * @param setpoint Target value
     * @param measured Current value
     * @param dt Time delta (seconds)
     * @return Control output
     */
    double computePID(double setpoint, double measured, double dt);
    
    /**
     * @brief Check safety conditions
     * @return True if safe to operate
     */
    bool checkSafety();
    
    /**
     * @brief Handle fault condition
     * @param state Fault state
     */
    void handleFault(MotorState state);
    
    /**
     * @brief Constrain value to range
     * @param value Value to constrain
     * @param min Minimum value
     * @param max Maximum value
     * @return Constrained value
     */
    double constrain(double value, double min, double max) const {
        return std::max(min, std::min(max, value));
    }
    
    // ========================================================================
    // PROTECTED MEMBER VARIABLES
    // ========================================================================
    
    MotorConfig config_;
    std::string name_;
    
    // State variables
    std::atomic<MotorState> current_state_;
    std::atomic<MotorDirection> current_direction_;
    std::atomic<double> current_velocity_;
    std::atomic<double> current_position_;
    std::atomic<double> current_current_;
    std::atomic<double> current_duty_cycle_;
    std::atomic<bool> is_enabled_;
    
    // Control variables
    double target_velocity_;
    double target_duty_cycle_;
    ControlMode control_mode_;
    
    // PID control
    PIDParameters pid_params_;
    double pid_integral_;
    double pid_last_error_;
    std::chrono::steady_clock::time_point pid_last_time_;
    
    // Encoder
    std::atomic<int64_t> encoder_count_;
    int64_t last_encoder_count_;
    std::chrono::steady_clock::time_point last_encoder_time_;
    
    // Velocity ramping
    double ramp_acceleration_;
    double ramp_deceleration_;
    double current_ramped_velocity_;
    
    // Thread safety
    mutable std::mutex state_mutex_;
    mutable std::mutex control_mutex_;
    
    // Callbacks
    std::function<void(MotorState)> fault_callback_;
    std::function<void(double)> overcurrent_callback_;
    
    // Statistics
    size_t fault_count_;
    size_t overcurrent_count_;
    std::chrono::steady_clock::time_point start_time_;
};

// ============================================================================
// L298N MOTOR DRIVER IMPLEMENTATION
// ============================================================================

/**
 * @brief Concrete implementation for L298N motor driver
 * 
 * L298N is a dual H-bridge motor driver IC supporting:
 * - 2 DC motors or 1 stepper motor
 * - Up to 2A per channel (4A peak)
 * - Operating voltage: 5-35V
 * - Built-in protection diodes
 */
class L298NDriver : public MotorDriver {
public:
    /**
     * @brief Constructor
     * @param config Motor configuration
     * @param name Motor name
     */
    explicit L298NDriver(const MotorConfig& config, const std::string& name = "L298N");
    
    /**
     * @brief Destructor
     */
    ~L298NDriver() override;
    
    /**
     * @brief Initialize L298N hardware
     * @return True if successful
     */
    bool initialize() override;

protected:
    /**
     * @brief Write PWM to L298N (ENA/ENB pin)
     * @param duty_cycle Duty cycle (0.0 - 1.0)
     */
    void writePWM(double duty_cycle) override;
    
    /**
     * @brief Set L298N direction (IN1/IN2 pins)
     * @param direction Direction to set
     */
    void setHardwareDirection(MotorDirection direction) override;

private:
    // Hardware interface pointers (platform-specific)
    std::shared_ptr<void> gpio_interface_;
    std::shared_ptr<void> pwm_interface_;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Convert motor state to string
 * @param state Motor state
 * @return String representation
 */
inline std::string toString(MotorState state) {
    switch (state) {
        case MotorState::IDLE: return "IDLE";
        case MotorState::RUNNING: return "RUNNING";
        case MotorState::BRAKING: return "BRAKING";
        case MotorState::FAULT: return "FAULT";
        case MotorState::OVERCURRENT: return "OVERCURRENT";
        case MotorState::OVERTEMPERATURE: return "OVERTEMPERATURE";
        case MotorState::STALLED: return "STALLED";
        case MotorState::DISABLED: return "DISABLED";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert motor direction to string
 * @param dir Motor direction
 * @return String representation
 */
inline std::string toString(MotorDirection dir) {
    switch (dir) {
        case MotorDirection::FORWARD: return "FORWARD";
        case MotorDirection::BACKWARD: return "BACKWARD";
        case MotorDirection::STOPPED: return "STOPPED";
        case MotorDirection::BRAKE: return "BRAKE";
        default: return "UNKNOWN";
    }
}

/**
 * @brief Convert RPM to rad/s
 * @param rpm Revolutions per minute
 * @return Radians per second
 */
inline double rpmToRadPerSec(double rpm) {
    return rpm * 2.0 * M_PI / 60.0;
}

/**
 * @brief Convert rad/s to RPM
 * @param rad_per_sec Radians per second
 * @return Revolutions per minute
 */
inline double radPerSecToRpm(double rad_per_sec) {
    return rad_per_sec * 60.0 / (2.0 * M_PI);
}

} // namespace robot_control

#endif // ROBOT_CONTROL_MOTOR_DRIVER_HPP_
