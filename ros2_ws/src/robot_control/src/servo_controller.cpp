#include "robot_control/servo_controller.hpp"
#include <pigpio.h>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>

namespace robot_control {

ServoController::ServoController(rclcpp::Node::SharedPtr node)
    : node_(node), 
      gpio_initialized_(false),
      emergency_stop_(false)
{
    RCLCPP_INFO(node_->get_logger(), "Initializing Servo Controller");
    
    // Declare parameters
    node_->declare_parameter("servo.pins", std::vector<int>{17, 18, 22, 23, 24, 25});
    node_->declare_parameter("servo.frequency", 50.0);
    node_->declare_parameter("servo.min_pulse", 500);
    node_->declare_parameter("servo.max_pulse", 2500);
    node_->declare_parameter("servo.min_angle", -90.0);
    node_->declare_parameter("servo.max_angle", 90.0);
    node_->declare_parameter("servo.safety_timeout", 2.0);
    node_->declare_parameter("servo.max_velocity", 180.0);  // degrees per second
    
    // Load calibration
    loadCalibration();
    
    // Initialize GPIO
    if (!initializeGPIO()) {
        RCLCPP_FATAL(node_->get_logger(), "Failed to initialize GPIO");
        throw std::runtime_error("GPIO initialization failed");
    }
    
    // Initialize servos
    initializeServos();
    
    // Create ROS2 interfaces
    initializeROSInterfaces();
    
    // Start control thread
    control_thread_ = std::thread(&ServoController::controlLoop, this);
    
    RCLCPP_INFO(node_->get_logger(), "Servo Controller initialized with %zu servos", 
                servo_configs_.size());
}

ServoController::~ServoController() {
    emergency_stop_ = true;
    
    if (control_thread_.joinable()) {
        control_thread_.join();
    }
    
    // Stop all servos
    for (const auto& config : servo_configs_) {
        gpioServo(config.pin, 0);  // Stop PWM
    }
    
    if (gpio_initialized_) {
        gpioTerminate();
    }
}

bool ServoController::initializeGPIO() {
    if (gpioInitialise() < 0) {
        RCLCPP_ERROR(node_->get_logger(), "pigpio initialization failed");
        return false;
    }
    
    gpio_initialized_ = true;
    gpioSetSignalFunc(SIGINT, ServoController::signalHandler);
    
    return true;
}

void ServoController::initializeServos() {
    auto pin_list = node_->get_parameter("servo.pins").as_integer_array();
    double frequency = node_->get_parameter("servo.frequency").as_double();
    int min_pulse = node_->get_parameter("servo.min_pulse").as_int();
    int max_pulse = node_->get_parameter("servo.max_pulse").as_int();
    double min_angle = node_->get_parameter("servo.min_angle").as_double();
    double max_angle = node_->get_parameter("servo.max_angle").as_double();
    double max_velocity = node_->get_parameter("servo.max_velocity").as_double();
    
    for (size_t i = 0; i < pin_list.size(); ++i) {
        ServoConfig config;
        config.pin = pin_list[i];
        config.frequency = frequency;
        config.min_pulse = min_pulse;
        config.max_pulse = max_pulse;
        config.min_angle = min_angle;
        config.max_angle = max_angle;
        config.max_velocity = max_velocity;
        config.current_angle = 0.0;
        config.target_angle = 0.0;
        config.current_pulse = angleToPulse(0.0, config);
        config.is_moving = false;
        config.last_update = std::chrono::steady_clock::now();
        
        // Set pin mode
        gpioSetMode(config.pin, PI_OUTPUT);
        
        // Initialize PWM
        gpioServo(config.pin, config.current_pulse);
        
        servo_configs_.push_back(config);
        
        RCLCPP_DEBUG(node_->get_logger(), 
                    "Initialized servo %zu on pin %d (pulse: %d)", 
                    i, config.pin, config.current_pulse);
    }
}

void ServoController::initializeROSInterfaces() {
    // Service for emergency stop
    emergency_stop_srv_ = node_->create_service<std_srvs::srv::Trigger>(
        "/servo/emergency_stop",
        std::bind(&ServoController::emergencyStopCallback, this,
                 std::placeholders::_1, std::placeholders::_2));
    
    // Service for reset
    reset_srv_ = node_->create_service<std_srvs::srv::Trigger>(
        "/servo/reset",
        std::bind(&ServoController::resetCallback, this,
                 std::placeholders::_1, std::placeholders::_2));
    
    // Action server for trajectory execution
    trajectory_action_server_ = rclcpp_action::create_server<FollowJointTrajectory>(
        node_,
        "/servo/follow_joint_trajectory",
        std::bind(&ServoController::handleTrajectoryGoal, this, std::placeholders::_1, std::placeholders::_2),
        std::bind(&ServoController::handleTrajectoryCancel, this, std::placeholders::_1),
        std::bind(&ServoController::handleTrajectoryAccepted, this, std::placeholders::_1));
    
    // Joint state publisher
    joint_state_pub_ = node_->create_publisher<sensor_msgs::msg::JointState>(
        "/joint_states", 10);
    
    // Command subscriber
    joint_command_sub_ = node_->create_subscription<trajectory_msgs::msg::JointTrajectory>(
        "/joint_trajectory_commands",
        10,
        std::bind(&ServoController::jointCommandCallback, this, std::placeholders::_1));
    
    RCLCPP_INFO(node_->get_logger(), "ROS2 interfaces initialized");
}

void ServoController::loadCalibration() {
    // Load calibration from YAML file
    std::string calibration_file;
    node_->declare_parameter("calibration_file", "/ros2_ws/src/robot_control/config/servo_calibration.yaml");
    node_->get_parameter("calibration_file", calibration_file);
    
    try {
        YAML::Node config = YAML::LoadFile(calibration_file);
        
        if (config["servo_calibration"]) {
            auto calibration = config["servo_calibration"];
            
            for (const auto& servo : calibration) {
                int servo_id = servo["id"].as<int>();
                double offset = servo["offset"].as<double>();
                double scale = servo["scale"].as<double>();
                bool inverted = servo["inverted"].as<bool>();
                
                servo_calibration_[servo_id] = {offset, scale, inverted};
                
                RCLCPP_INFO(node_->get_logger(), 
                          "Loaded calibration for servo %d: offset=%.3f, scale=%.3f, inverted=%d",
                          servo_id, offset, scale, inverted);
            }
        }
    } catch (const YAML::Exception& e) {
        RCLCPP_WARN(node_->get_logger(), 
                   "Failed to load calibration file %s: %s. Using defaults.",
                   calibration_file.c_str(), e.what());
    }
}

int ServoController::angleToPulse(double angle, const ServoConfig& config) {
    // Apply calibration if available
    double calibrated_angle = angle;
    if (servo_calibration_.find(config.pin) != servo_calibration_.end()) {
        const auto& cal = servo_calibration_[config.pin];
        calibrated_angle = (calibrated_angle + cal.offset) * cal.scale;
        if (cal.inverted) {
            calibrated_angle = -calibrated_angle;
        }
    }
    
    // Clamp to limits
    calibrated_angle = std::clamp(calibrated_angle, config.min_angle, config.max_angle);
    
    // Convert angle to pulse width
    double normalized = (calibrated_angle - config.min_angle) / 
                       (config.max_angle - config.min_angle);
    int pulse = static_cast<int>(config.min_pulse + 
                                normalized * (config.max_pulse - config.min_pulse));
    
    return std::clamp(pulse, config.min_pulse, config.max_pulse);
}

double ServoController::pulseToAngle(int pulse, const ServoConfig& config) {
    // Convert pulse to normalized angle
    double normalized = static_cast<double>(pulse - config.min_pulse) / 
                       (config.max_pulse - config.min_pulse);
    double angle = config.min_angle + normalized * (config.max_angle - config.min_angle);
    
    // Apply inverse calibration
    if (servo_calibration_.find(config.pin) != servo_calibration_.end()) {
        const auto& cal = servo_calibration_[config.pin];
        if (cal.inverted) {
            angle = -angle;
        }
        angle = angle / cal.scale - cal.offset;
    }
    
    return angle;
}

bool ServoController::setAngle(int servo_index, double angle, bool immediate) {
    if (servo_index < 0 || servo_index >= static_cast<int>(servo_configs_.size())) {
        RCLCPP_ERROR(node_->get_logger(), "Invalid servo index: %d", servo_index);
        return false;
    }
    
    if (emergency_stop_) {
        RCLCPP_WARN(node_->get_logger(), "Emergency stop active, ignoring command");
        return false;
    }
    
    auto& config = servo_configs_[servo_index];
    
    // Check velocity limits
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - config.last_update).count();
    
    if (dt > 0) {
        double requested_velocity = std::abs(angle - config.current_angle) / dt;
        if (requested_velocity > config.max_velocity) {
            RCLCPP_WARN(node_->get_logger(), 
                       "Servo %d velocity limit exceeded: %.1f > %.1f deg/s",
                       servo_index, requested_velocity, config.max_velocity);
            
            if (!immediate) {
                // Scale movement to respect velocity limit
                double max_move = config.max_velocity * dt;
                if (angle > config.current_angle) {
                    angle = config.current_angle + max_move;
                } else {
                    angle = config.current_angle - max_move;
                }
            }
        }
    }
    
    // Update target
    config.target_angle = std::clamp(angle, config.min_angle, config.max_angle);
    config.is_moving = true;
    config.last_update = now;
    
    if (immediate) {
        return moveToTargetImmediate(servo_index);
    }
    
    return true;
}

bool ServoController::moveToTargetImmediate(int servo_index) {
    auto& config = servo_configs_[servo_index];
    
    int target_pulse = angleToPulse(config.target_angle, config);
    
    if (gpioServo(config.pin, target_pulse) == 0) {
        config.current_pulse = target_pulse;
        config.current_angle = pulseToAngle(target_pulse, config);
        config.is_moving = false;
        return true;
    }
    
    RCLCPP_ERROR(node_->get_logger(), "Failed to set servo %d to pulse %d", 
                servo_index, target_pulse);
    return false;
}

void ServoController::controlLoop() {
    rclcpp::Rate rate(100);  // 100 Hz control loop
    
    while (rclcpp::ok() && !emergency_stop_) {
        auto now = std::chrono::steady_clock::now();
        
        for (size_t i = 0; i < servo_configs_.size(); ++i) {
            auto& config = servo_configs_[i];
            
            if (config.is_moving) {
                double dt = std::chrono::duration<double>(now - config.last_update).count();
                
                // Check for timeout
                if (dt > node_->get_parameter("servo.safety_timeout").as_double()) {
                    RCLCPP_WARN(node_->get_logger(), 
                               "Servo %zu movement timeout, stopping", i);
                    config.is_moving = false;
                    continue;
                }
                
                // Calculate smooth movement
                double angle_diff = config.target_angle - config.current_angle;
                double max_move = config.max_velocity * dt;
                
                if (std::abs(angle_diff) <= max_move) {
                    // Reached target
                    config.current_angle = config.target_angle;
                    config.is_moving = false;
                } else {
                    // Move towards target
                    config.current_angle += (angle_diff > 0 ? max_move : -max_move);
                }
                
                // Update servo
                int pulse = angleToPulse(config.current_angle, config);
                if (gpioServo(config.pin, pulse) == 0) {
                    config.current_pulse = pulse;
                } else {
                    RCLCPP_ERROR(node_->get_logger(), 
                               "Failed to update servo %zu to pulse %d", i, pulse);
                }
                
                config.last_update = now;
            }
        }
        
        // Publish joint states
        publishJointStates();
        
        rate.sleep();
    }
}

void ServoController::publishJointStates() {
    auto msg = sensor_msgs::msg::JointState();
    msg.header.stamp = node_->now();
    
    for (size_t i = 0; i < servo_configs_.size(); ++i) {
        const auto& config = servo_configs_[i];
        msg.name.push_back("servo_" + std::to_string(i));
        msg.position.push_back(config.current_angle * M_PI / 180.0);  // Convert to radians
        msg.velocity.push_back(0.0);  // Would need to calculate actual velocity
        msg.effort.push_back(0.0);
    }
    
    joint_state_pub_->publish(msg);
}

void ServoController::jointCommandCallback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
    if (emergency_stop_) {
        RCLCPP_WARN(node_->get_logger(), "Emergency stop active, ignoring trajectory");
        return;
    }
    
    RCLCPP_INFO(node_->get_logger(), "Received trajectory with %zu points", 
                msg->points.size());
    
    // Simple implementation - execute first point immediately
    // In production, this would implement full trajectory following
    if (!msg->points.empty() && !msg->joint_names.empty()) {
        const auto& point = msg->points[0];
        
        for (size_t i = 0; i < msg->joint_names.size() && i < point.positions.size(); ++i) {
            // Find servo index by name
            // This assumes naming convention: servo_0, servo_1, etc.
            try {
                size_t servo_index = std::stoi(msg->joint_names[i].substr(6));
                if (servo_index < servo_configs_.size()) {
                    double angle_rad = point.positions[i];
                    double angle_deg = angle_rad * 180.0 / M_PI;
                    setAngle(servo_index, angle_deg, true);
                }
            } catch (...) {
                RCLCPP_WARN(node_->get_logger(), "Invalid joint name: %s", 
                           msg->joint_names[i].c_str());
            }
        }
    }
}

rclcpp_action::GoalResponse ServoController::handleTrajectoryGoal(
    const rclcpp_action::GoalUUID& uuid,
    std::shared_ptr<const FollowJointTrajectory::Goal> goal)
{
    RCLCPP_INFO(node_->get_logger(), "Received trajectory action goal");
    (void)uuid;
    
    if (emergency_stop_) {
        RCLCPP_WARN(node_->get_logger(), "Emergency stop active, rejecting goal");
        return rclcpp_action::GoalResponse::REJECT;
    }
    
    // Validate trajectory
    if (goal->trajectory.points.empty()) {
        RCLCPP_WARN(node_->get_logger(), "Empty trajectory, rejecting");
        return rclcpp_action::GoalResponse::REJECT;
    }
    
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse ServoController::handleTrajectoryCancel(
    const std::shared_ptr<GoalHandleFollowJointTrajectory> goal_handle)
{
    RCLCPP_INFO(node_->get_logger(), "Received trajectory cancellation");
    (void)goal_handle;
    return rclcpp_action::CancelResponse::ACCEPT;
}

void ServoController::handleTrajectoryAccepted(
    const std::shared_ptr<GoalHandleFollowJointTrajectory> goal_handle)
{
    // Execute trajectory in separate thread
    std::thread([this, goal_handle]() {
        executeTrajectory(goal_handle);
    }).detach();
}

void ServoController::executeTrajectory(
    const std::shared_ptr<GoalHandleFollowJointTrajectory> goal_handle)
{
    auto result = std::make_shared<FollowJointTrajectory::Result>();
    auto feedback = std::make_shared<FollowJointTrajectory::Feedback>();
    
    const auto& trajectory = goal_handle->get_goal()->trajectory;
    
    try {
        auto start_time = node_->now();
        
        for (size_t i = 0; i < trajectory.points.size(); ++i) {
            if (emergency_stop_) {
                throw std::runtime_error("Emergency stop triggered");
            }
            
            if (goal_handle->is_canceling()) {
                result->error_code = FollowJointTrajectory::Result::SUCCESSFUL;
                goal_handle->canceled(result);
                return;
            }
            
            const auto& point = trajectory.points[i];
            
            // Execute point
            for (size_t j = 0; j < trajectory.joint_names.size() && j < point.positions.size(); ++j) {
                try {
                    size_t servo_index = std::stoi(trajectory.joint_names[j].substr(6));
                    if (servo_index < servo_configs_.size()) {
                        double angle_rad = point.positions[j];
                        double angle_deg = angle_rad * 180.0 / M_PI;
                        setAngle(servo_index, angle_deg, true);
                    }
                } catch (...) {
                    // Skip invalid joints
                }
            }
            
            // Publish feedback
            if (i % 10 == 0) {  // Every 10th point
                feedback->header.stamp = node_->now();
                feedback->joint_names = trajectory.joint_names;
                feedback->desired = point;
                feedback->actual = point;  // Simplified
                feedback->error.positions.resize(point.positions.size(), 0.0);
                
                goal_handle->publish_feedback(feedback);
            }
            
            // Sleep for time_from_start
            if (i + 1 < trajectory.points.size()) {
                auto next_time = trajectory.points[i + 1].time_from_start;
                auto current_time = point.time_from_start;
                auto sleep_duration = next_time - current_time;
                
                rclcpp::sleep_for(std::chrono::duration_cast<std::chrono::nanoseconds>(sleep_duration));
            }
        }
        
        // Success
        result->error_code = FollowJointTrajectory::Result::SUCCESSFUL;
        goal_handle->succeed(result);
        
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Trajectory execution failed: %s", e.what());
        result->error_code = FollowJointTrajectory::Result::INVALID_GOAL;
        result->error_string = e.what();
        goal_handle->abort(result);
    }
}

void ServoController::emergencyStopCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    (void)request;
    
    emergency_stop_ = true;
    
    // Stop all servos
    for (const auto& config : servo_configs_) {
        gpioServo(config.pin, 0);
    }
    
    response->success = true;
    response->message = "Emergency stop activated";
    
    RCLCPP_WARN(node_->get_logger(), "EMERGENCY STOP ACTIVATED");
}

void ServoController::resetCallback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    (void)request;
    
    if (!emergency_stop_) {
        response->success = false;
        response->message = "Emergency stop not active";
        return;
    }
    
    emergency_stop_ = false;
    
    // Reset to home position
    for (size_t i = 0; i < servo_configs_.size(); ++i) {
        setAngle(i, 0.0, true);
    }
    
    response->success = true;
    response->message = "System reset to home position";
    
    RCLCPP_INFO(node_->get_logger(), "System reset complete");
}

void ServoController::signalHandler(int signum) {
    (void)signum;
    RCLCPP_WARN(rclcpp::get_logger("servo_controller"), "Received signal, shutting down");
    gpioTerminate();
}

}  // namespace robot_control
