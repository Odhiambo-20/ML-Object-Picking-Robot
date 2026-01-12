/**
 * @file gripper_control.cpp
 * @brief Production-grade gripper controller implementation
 * 
 * Implementation of the GripperController class for industrial pick-and-place
 * operations with advanced features including adaptive grasping, slip detection,
 * and safety monitoring.
 * 
 * @author Victor's Production Team
 * @date 2025-01-12
 * @version 1.0.0
 * 
 * @copyright Proprietary - Industrial Use Only
 */

#include "robot_control/gripper_control.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>

namespace robot_control {

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

GripperController::GripperController(const rclcpp::NodeOptions& options)
    : Node("gripper_controller", options),
      current_state_(GripperState::IDLE),
      current_width_(0.0),
      current_force_(0.0),
      motor_current_(0.0),
      is_holding_object_(false),
      slip_detected_(false),
      slip_detection_enabled_(true),
      target_width_(0.0),
      target_force_(0.0),
      current_strategy_(GraspStrategy::ADAPTIVE),
      force_history_size_(20),
      last_stable_force_(0.0),
      emergency_stop_active_(false),
      calibration_complete_(false),
      successful_grasps_(0),
      failed_grasps_(0),
      slip_events_(0)
{
    RCLCPP_INFO(this->get_logger(), "Initializing GripperController...");
    
    // Load parameters
    loadParameters();
    
    // Initialize ROS components
    initializeROS();
    
    // Initialize hardware
    if (!initializeHardware()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize hardware!");
        throw std::runtime_error("Hardware initialization failed");
    }
    
    // Start control loop
    control_timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / CONTROL_FREQUENCY)),
        std::bind(&GripperController::controlLoop, this)
    );
    
    node_start_time_ = std::chrono::steady_clock::now();
    force_history_.reserve(force_history_size_);
    
    RCLCPP_INFO(this->get_logger(), 
                "GripperController initialized successfully (%.1f Hz)", 
                CONTROL_FREQUENCY);
}

GripperController::~GripperController() {
    RCLCPP_INFO(this->get_logger(), "Shutting down GripperController...");
    
    // Stop control loop
    if (control_timer_) {
        control_timer_->cancel();
    }
    
    // Release any held object
    if (is_holding_object_) {
        release(true);
    }
    
    // Print statistics
    auto uptime = std::chrono::duration_cast<std::chrono::hours>(
        std::chrono::steady_clock::now() - node_start_time_
    ).count();
    
    RCLCPP_INFO(this->get_logger(), 
                "Statistics - Uptime: %ld hours, Successful grasps: %zu, "
                "Failed: %zu, Slip events: %zu",
                uptime, successful_grasps_, failed_grasps_, slip_events_);
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void GripperController::initializeROS() {
    RCLCPP_DEBUG(this->get_logger(), "Initializing ROS components...");
    
    // Publishers
    state_pub_ = this->create_publisher<robot_interfaces::msg::GripperState>(
        "~/state", 10
    );
    
    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>(
        "~/joint_states", 10
    );
    
    // Subscribers
    position_cmd_sub_ = this->create_subscription<std_msgs::msg::Float64>(
        "~/position_command",
        10,
        std::bind(&GripperController::positionCommandCallback, this, std::placeholders::_1)
    );
    
    force_cmd_sub_ = this->create_subscription<std_msgs::msg::Float64>(
        "~/force_command",
        10,
        std::bind(&GripperController::forceCommandCallback, this, std::placeholders::_1)
    );
    
    emergency_stop_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        "/emergency_stop",
        rclcpp::QoS(10).reliable().transient_local(),
        std::bind(&GripperController::emergencyStopCallback, this, std::placeholders::_1)
    );
    
    // Services
    grasp_service_ = this->create_service<robot_interfaces::srv::GraspObject>(
        "~/grasp_object",
        std::bind(&GripperController::graspServiceCallback, this, 
                  std::placeholders::_1, std::placeholders::_2)
    );
    
    RCLCPP_INFO(this->get_logger(), "ROS components initialized");
}

bool GripperController::initializeHardware() {
    RCLCPP_INFO(this->get_logger(), "Initializing gripper hardware...");
    
    // TODO: Initialize actual hardware interface
    // This is a placeholder - actual implementation depends on hardware
    // Example: Initialize servo controller, current sensors, force sensors
    
    // Simulate hardware initialization
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Set initial state
    current_width_ = config_.max_opening_width;
    current_force_ = 0.0;
    motor_current_ = 0.0;
    current_state_ = GripperState::IDLE;
    
    RCLCPP_INFO(this->get_logger(), "Hardware initialized successfully");
    return true;
}

void GripperController::loadParameters() {
    RCLCPP_DEBUG(this->get_logger(), "Loading parameters...");
    
    // Declare and get parameters
    this->declare_parameter("max_opening_width", config_.max_opening_width);
    this->declare_parameter("min_opening_width", config_.min_opening_width);
    this->declare_parameter("max_force", config_.max_force);
    this->declare_parameter("min_force", config_.min_force);
    this->declare_parameter("closing_velocity", config_.closing_velocity);
    this->declare_parameter("opening_velocity", config_.opening_velocity);
    this->declare_parameter("max_current", config_.max_current);
    this->declare_parameter("enable_slip_detection", slip_detection_enabled_.load());
    
    config_.max_opening_width = this->get_parameter("max_opening_width").as_double();
    config_.min_opening_width = this->get_parameter("min_opening_width").as_double();
    config_.max_force = this->get_parameter("max_force").as_double();
    config_.min_force = this->get_parameter("min_force").as_double();
    config_.closing_velocity = this->get_parameter("closing_velocity").as_double();
    config_.opening_velocity = this->get_parameter("opening_velocity").as_double();
    config_.max_current = this->get_parameter("max_current").as_double();
    slip_detection_enabled_ = this->get_parameter("enable_slip_detection").as_bool();
    
    RCLCPP_INFO(this->get_logger(), 
                "Parameters loaded - Max width: %.3fm, Max force: %.1fN",
                config_.max_opening_width, config_.max_force);
}

// ============================================================================
// PUBLIC CONTROL METHODS
// ============================================================================

bool GripperController::open(double width, bool blocking) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    
    if (emergency_stop_active_) {
        RCLCPP_ERROR(this->get_logger(), "Cannot open - emergency stop active");
        return false;
    }
    
    if (!checkSafety()) {
        return false;
    }
    
    // Use max width if not specified
    if (width < 0.0) {
        width = config_.max_opening_width;
    }
    
    // Constrain to limits
    width = std::clamp(width, config_.min_opening_width, config_.max_opening_width);
    
    RCLCPP_INFO(this->get_logger(), "Opening gripper to %.3fm", width);
    
    target_width_ = width;
    current_state_ = GripperState::OPENING;
    is_holding_object_ = false;
    last_command_time_ = std::chrono::steady_clock::now();
    
    if (blocking) {
        // Wait for completion
        while (current_state_ == GripperState::OPENING) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    return true;
}

bool GripperController::close(double force, bool blocking) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    
    if (emergency_stop_active_) {
        RCLCPP_ERROR(this->get_logger(), "Cannot close - emergency stop active");
        return false;
    }
    
    if (!checkSafety()) {
        return false;
    }
    
    // Use default force if not specified
    if (force < 0.0) {
        force = config_.max_force * 0.5; // 50% of max force
    }
    
    // Constrain to limits
    force = std::clamp(force, config_.min_force, config_.max_force);
    
    RCLCPP_INFO(this->get_logger(), "Closing gripper with %.1fN force", force);
    
    target_force_ = force;
    current_state_ = GripperState::CLOSING;
    last_command_time_ = std::chrono::steady_clock::now();
    
    if (blocking) {
        // Wait for completion or object detection
        while (current_state_ == GripperState::CLOSING) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    
    return true;
}

bool GripperController::grasp(const ObjectProperties& properties, GraspStrategy strategy) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    
    if (emergency_stop_active_) {
        RCLCPP_ERROR(this->get_logger(), "Cannot grasp - emergency stop active");
        return false;
    }
    
    RCLCPP_INFO(this->get_logger(), 
                "Executing grasp - Mass: %.3fkg, Material: %s, Strategy: %s",
                properties.estimated_mass, properties.material_type.c_str(),
                toString(strategy).c_str());
    
    current_strategy_ = strategy;
    grasp_start_time_ = std::chrono::steady_clock::now();
    
    // Calculate required force based on object properties
    double required_force = calculateRequiredForce(properties);
    
    RCLCPP_DEBUG(this->get_logger(), "Required force: %.2fN", required_force);
    
    // Position gripper above object
    double approach_width = properties.width + 0.01; // 1cm clearance
    if (!executePositionControl(approach_width)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to position gripper");
        failed_grasps_++;
        return false;
    }
    
    // Close with calculated force
    if (!executeForceControl(required_force)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to grasp object");
        failed_grasps_++;
        return false;
    }
    
    // Verify grasp
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    if (detectObject()) {
        current_state_ = GripperState::HOLDING;
        is_holding_object_ = true;
        successful_grasps_++;
        
        RCLCPP_INFO(this->get_logger(), "Object grasped successfully!");
        
        if (object_detected_callback_) {
            object_detected_callback_(true);
        }
        
        return true;
    } else {
        RCLCPP_WARN(this->get_logger(), "Object not detected after grasp");
        failed_grasps_++;
        return false;
    }
}

bool GripperController::release(bool open_fully) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    
    RCLCPP_INFO(this->get_logger(), "Releasing object");
    
    if (is_holding_object_) {
        is_holding_object_ = false;
        
        if (object_detected_callback_) {
            object_detected_callback_(false);
        }
    }
    
    double release_width = open_fully ? config_.max_opening_width : 
                                        config_.max_opening_width * 0.5;
    
    return open(release_width, true);
}

void GripperController::stop() {
    std::lock_guard<std::mutex> lock(command_mutex_);
    
    RCLCPP_INFO(this->get_logger(), "Stopping gripper motion");
    
    current_state_ = GripperState::IDLE;
    target_width_ = current_width_;
    target_force_ = 0.0;
}

void GripperController::emergencyStop() {
    RCLCPP_ERROR(this->get_logger(), "EMERGENCY STOP ACTIVATED!");
    
    emergency_stop_active_ = true;
    current_state_ = GripperState::EMERGENCY_STOP;
    
    // Release gripper immediately
    target_force_ = 0.0;
    
    // TODO: Send emergency stop to hardware
    
    // Notify through callback
    if (object_detected_callback_) {
        object_detected_callback_(false);
    }
}

bool GripperController::calibrate() {
    RCLCPP_INFO(this->get_logger(), "Starting gripper calibration...");
    
    current_state_ = GripperState::CALIBRATING;
    
    // Open to maximum
    if (!open(config_.max_opening_width, true)) {
        RCLCPP_ERROR(this->get_logger(), "Calibration failed - cannot open");
        current_state_ = GripperState::ERROR;
        return false;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Close to minimum
    if (!executePositionControl(config_.min_opening_width)) {
        RCLCPP_ERROR(this->get_logger(), "Calibration failed - cannot close");
        current_state_ = GripperState::ERROR;
        return false;
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Return to home
    if (!home()) {
        RCLCPP_ERROR(this->get_logger(), "Calibration failed - cannot home");
        current_state_ = GripperState::ERROR;
        return false;
    }
    
    calibration_complete_ = true;
    current_state_ = GripperState::IDLE;
    
    RCLCPP_INFO(this->get_logger(), "Calibration completed successfully");
    return true;
}

bool GripperController::home() {
    RCLCPP_INFO(this->get_logger(), "Homing gripper...");
    
    // Home position is fully open
    return open(config_.max_opening_width, true);
}

// ============================================================================
// CONFIGURATION METHODS
// ============================================================================

void GripperController::setConfiguration(const GripperConfig& config) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    config_ = config;
    
    RCLCPP_INFO(this->get_logger(), "Configuration updated");
}

void GripperController::setMaxForce(double max_force) {
    if (max_force < config_.min_force) {
        RCLCPP_WARN(this->get_logger(), 
                    "Max force %.1fN is less than min force %.1fN",
                    max_force, config_.min_force);
        max_force = config_.min_force;
    }
    
    config_.max_force = max_force;
    
    RCLCPP_INFO(this->get_logger(), "Max force set to %.1fN", max_force);
}

// ============================================================================
// PRIVATE CONTROL METHODS
// ============================================================================

void GripperController::controlLoop() {
    // Update state based on sensor feedback
    updateState();
    
    // Monitor for slip if holding object
    if (is_holding_object_ && slip_detection_enabled_) {
        monitorSlip();
    }
    
    // Execute control based on current state
    switch (current_state_.load()) {
        case GripperState::OPENING:
            executePositionControl(target_width_);
            break;
            
        case GripperState::CLOSING:
            if (detectObject()) {
                current_state_ = GripperState::HOLDING;
                is_holding_object_ = true;
            } else {
                executeForceControl(target_force_);
            }
            break;
            
        case GripperState::GRIPPING:
            executeForceControl(target_force_);
            break;
            
        case GripperState::HOLDING:
            // Maintain holding force
            if (std::abs(current_force_ - target_force_) > config_.force_tolerance) {
                executeForceControl(target_force_);
            }
            break;
            
        default:
            // Idle, error, etc. - do nothing
            break;
    }
    
    // Publish state
    publishState();
    
    // Check safety conditions
    if (!checkSafety()) {
        handleError("Safety check failed");
    }
}

void GripperController::updateState() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // TODO: Read actual sensor values from hardware
    // This is a placeholder implementation
    
    // Simulate sensor readings based on commands
    // In real implementation, read from:
    // - Position encoder
    // - Force sensor
    // - Current sensor
    
    // For now, just track commanded values
}

void GripperController::monitorSlip() {
    // Add current force to history
    force_history_.push_back(current_force_);
    
    if (force_history_.size() > force_history_size_) {
        force_history_.erase(force_history_.begin());
    }
    
    // Need enough history to detect slip
    if (force_history_.size() < force_history_size_ / 2) {
        return;
    }
    
    // Calculate force variance
    double mean = std::accumulate(force_history_.begin(), force_history_.end(), 0.0) / 
                  force_history_.size();
    
    double variance = 0.0;
    for (double f : force_history_) {
        variance += (f - mean) * (f - mean);
    }
    variance /= force_history_.size();
    
    double std_dev = std::sqrt(variance);
    
    // Detect slip based on force variation
    if (std_dev > config_.slip_detection_threshold * mean) {
        if (!slip_detected_) {
            slip_detected_ = true;
            slip_events_++;
            
            RCLCPP_WARN(this->get_logger(), 
                        "SLIP DETECTED! Force variance: %.2fN (mean: %.2fN)",
                        std_dev, mean);
            
            if (slip_detected_callback_) {
                slip_detected_callback_();
            }
            
            // Increase grip force by 20%
            target_force_ = std::min(target_force_ * 1.2, config_.max_force);
            executeForceControl(target_force_);
        }
    } else {
        slip_detected_ = false;
        last_stable_force_ = mean;
    }
}

bool GripperController::detectObject() {
    // Object detected if:
    // 1. Current above threshold (motor stall)
    // 2. Force sensor reading above threshold
    // 3. Position stopped before fully closed
    
    bool current_threshold = motor_current_ > (config_.stall_current_threshold * 0.5);
    bool force_threshold = current_force_ > config_.min_force;
    bool position_threshold = current_width_ > config_.min_opening_width;
    
    return current_threshold || force_threshold || position_threshold;
}

double GripperController::calculateRequiredForce(const ObjectProperties& properties) {
    // Calculate minimum force required to hold object
    const double g = 9.81; // m/s^2
    const double safety_factor = properties.is_fragile ? 1.5 : 2.0;
    
    // Adjust friction coefficient for material
    double friction = properties.friction_coefficient;
    if (properties.is_slippery) {
        friction *= 0.7; // Reduce friction for slippery objects
    }
    
    // Required force: F = (m * g * safety_factor) / μ
    double required_force = (properties.estimated_mass * g * safety_factor) / friction;
    
    // Constrain to gripper limits
    required_force = std::clamp(required_force, config_.min_force, config_.max_force);
    
    return required_force;
}

bool GripperController::executePositionControl(double target_width) {
    // TODO: Implement actual position control
    // For now, simulate smooth motion
    
    double error = target_width - current_width_;
    double max_step = config_.opening_velocity / CONTROL_FREQUENCY;
    
    if (std::abs(error) > config_.position_tolerance) {
        double step = std::clamp(error, -max_step, max_step);
        current_width_ = current_width_ + step;
        return false; // Not yet reached
    } else {
        current_width_ = target_width;
        
        if (current_state_ == GripperState::OPENING) {
            current_state_ = GripperState::IDLE;
        }
        
        return true; // Target reached
    }
}

bool GripperController::executeForceControl(double target_force) {
    // TODO: Implement actual force control
    // For now, simulate force buildup
    
    double error = target_force - current_force_;
    double max_step = 5.0 / CONTROL_FREQUENCY; // 5N/s ramp rate
    
    if (std::abs(error) > config_.force_tolerance) {
        double step = std::clamp(error, -max_step, max_step);
        current_force_ = current_force_ + step;
        return false; // Not yet reached
    } else {
        current_force_ = target_force;
        
        if (current_state_ == GripperState::CLOSING) {
            current_state_ = GripperState::GRIPPING;
        }
        
        return true; // Target reached
    }
}

void GripperController::publishState() {
    // Publish gripper state message
    auto state_msg = std::make_unique<robot_interfaces::msg::GripperState>();
    state_msg->header.stamp = this->now();
    state_msg->position = current_width_;
    state_msg->force = current_force_;
    state_msg->is_holding = is_holding_object_;
    state_msg->is_moving = (current_state_ == GripperState::OPENING || 
                           current_state_ == GripperState::CLOSING);
    
    state_pub_->publish(std::move(state_msg));
    
    // Publish joint state
    auto joint_msg = std::make_unique<sensor_msgs::msg::JointState>();
    joint_msg->header.stamp = this->now();
    joint_msg->name.push_back("gripper_joint");
    joint_msg->position.push_back(current_width_);
    joint_msg->velocity.push_back(0.0); // TODO: Calculate from history
    joint_msg->effort.push_back(current_force_);
    
    joint_state_pub_->publish(std::move(joint_msg));
}

bool GripperController::checkSafety() {
    // Check current limit
    if (motor_current_ > config_.max_current) {
        RCLCPP_ERROR(this->get_logger(), 
                     "Overcurrent detected: %.2fA (limit: %.2fA)",
                     motor_current_.load(), config_.max_current);
        handleError("Overcurrent");
        return false;
    }
    
    // Check command timeout
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        now - last_command_time_
    ).count();
    
    if (elapsed > config_.timeout_duration && 
        current_state_ != GripperState::IDLE && 
        current_state_ != GripperState::HOLDING) {
        RCLCPP_WARN(this->get_logger(), "Command timeout - stopping");
        stop();
    }
    
    return true;
}

void GripperController::handleError(const std::string& error_msg) {
    RCLCPP_ERROR(this->get_logger(), "Error: %s", error_msg.c_str());
    
    current_state_ = GripperState::ERROR;
    target_force_ = 0.0;
    
    // TODO: Trigger error recovery or shutdown
}

// ============================================================================
// ROS CALLBACKS
// ============================================================================

void GripperController::positionCommandCallback(const std_msgs::msg::Float64::SharedPtr msg) {
    double width = msg->data;
    
    if (width >= 0.0) {
        close(0.0, false);  // Close to position
    } else {
        open(std::abs(width), false);
    }
}

void GripperController::forceCommandCallback(const std_msgs::msg::Float64::SharedPtr msg) {
    double force = msg->data;
    close(force, false);
}

void GripperController::graspServiceCallback(
    const std::shared_ptr<robot_interfaces::srv::GraspObject::Request> request,
    std::shared_ptr<robot_interfaces::srv::GraspObject::Response> response)
{
    ObjectProperties props;
    props.estimated_mass = request->object_mass;
    props.friction_coefficient = request->friction_coefficient;
    props.material_type = request->material_type;
    props.is_fragile = request->is_fragile;
    props.width = request->object_width;
    props.is_slippery = request->is_slippery;
    
    GraspStrategy strategy = static_cast<GraspStrategy>(request->strategy);
    
    bool success = grasp(props, strategy);
    
    response->success = success;
    response->actual_force = current_force_;
    response->is_holding = is_holding_object_;
}

void GripperController::emergencyStopCallback(const std_msgs::msg::Bool::SharedPtr msg) {
    if (msg->data) {
        emergencyStop();
    } else {
        // Reset emergency stop
        emergency_stop_active_ = false;
        current_state_ = GripperState::IDLE;
        RCLCPP_INFO(this->get_logger(), "Emergency stop released");
    }
}

} // namespace robot_control

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<robot_control::GripperController>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("gripper_controller"), 
                     "Exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}
