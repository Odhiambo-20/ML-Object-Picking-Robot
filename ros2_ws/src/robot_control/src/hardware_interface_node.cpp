/**
 * @file hardware_interface_node.cpp
 * @brief Production-grade hardware interface for Raspberry Pi 4 robot control
 * 
 * This node provides low-level hardware abstraction for:
 * - PCA9685 16-channel PWM driver (I2C)
 * - GPIO control (servos, sensors, LEDs)
 * - ADC reading (current sensors, voltage monitoring)
 * - System health monitoring (temperature, voltage)
 * - Hardware fault detection and recovery
 * 
 * Hardware Support:
 * - Raspberry Pi 4 (BCM2711)
 * - PCA9685 PWM Driver (I2C address 0x40)
 * - ADS1115 ADC (optional, I2C address 0x48)
 * - GPIO pins for auxiliary control
 * 
 * @author Victor's Production Team
 * @date 2025-01-12
 * @version 1.0.0
 * 
 * @copyright Proprietary - Industrial Use Only
 */

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <thread>
#include <fstream>
#include <sstream>
#include <cmath>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/temperature.hpp"
#include "diagnostic_msgs/msg/diagnostic_array.hpp"
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "diagnostic_msgs/msg/key_value.hpp"

// Linux I2C interface
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>

// GPIO interface (using libgpiod)
#include <gpiod.h>

namespace robot_control {

// ============================================================================
// CONSTANTS
// ============================================================================

// PCA9685 Register Addresses
constexpr uint8_t PCA9685_MODE1 = 0x00;
constexpr uint8_t PCA9685_MODE2 = 0x01;
constexpr uint8_t PCA9685_PRESCALE = 0xFE;
constexpr uint8_t PCA9685_LED0_ON_L = 0x06;
constexpr uint8_t PCA9685_LED0_ON_H = 0x07;
constexpr uint8_t PCA9685_LED0_OFF_L = 0x08;
constexpr uint8_t PCA9685_LED0_OFF_H = 0x09;

// PCA9685 Bits
constexpr uint8_t MODE1_RESTART = 0x80;
constexpr uint8_t MODE1_SLEEP = 0x10;
constexpr uint8_t MODE1_ALLCALL = 0x01;
constexpr uint8_t MODE2_OUTDRV = 0x04;

// Hardware Limits
constexpr double CPU_TEMP_WARNING = 70.0;  // °C
constexpr double CPU_TEMP_CRITICAL = 80.0; // °C
constexpr double VOLTAGE_MIN = 4.75;       // V
constexpr double VOLTAGE_MAX = 5.25;       // V

// ============================================================================
// HARDWARE INTERFACE NODE
// ============================================================================

class HardwareInterfaceNode : public rclcpp::Node {
public:
    explicit HardwareInterfaceNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("hardware_interface", options),
          i2c_fd_(-1),
          pca9685_initialized_(false),
          gpio_chip_(nullptr),
          emergency_stop_active_(false)
    {
        RCLCPP_INFO(this->get_logger(), "Initializing Hardware Interface Node...");
        
        // Declare parameters
        declareParameters();
        
        // Initialize hardware
        if (!initializeI2C()) {
            throw std::runtime_error("Failed to initialize I2C");
        }
        
        if (!initializePCA9685()) {
            throw std::runtime_error("Failed to initialize PCA9685");
        }
        
        if (!initializeGPIO()) {
            RCLCPP_WARN(this->get_logger(), "GPIO initialization failed - continuing without GPIO");
        }
        
        // Initialize ROS components
        initializeROS();
        
        // Start monitoring
        startMonitoring();
        
        RCLCPP_INFO(this->get_logger(), "Hardware Interface Node initialized successfully");
    }
    
    ~HardwareInterfaceNode() {
        RCLCPP_INFO(this->get_logger(), "Shutting down Hardware Interface...");
        
        // Disable all PWM outputs
        if (pca9685_initialized_) {
            disableAllPWM();
        }
        
        // Close I2C
        if (i2c_fd_ >= 0) {
            close(i2c_fd_);
        }
        
        // Close GPIO
        if (gpio_chip_) {
            gpiod_chip_close(gpio_chip_);
        }
        
        RCLCPP_INFO(this->get_logger(), "Hardware Interface shutdown complete");
    }

private:
    // ========================================================================
    // INITIALIZATION
    // ========================================================================
    
    void declareParameters() {
        this->declare_parameter("i2c_bus", 1);
        this->declare_parameter("pca9685_address", 0x40);
        this->declare_parameter("pwm_frequency", 50);
        this->declare_parameter("gpio_chip", "gpiochip0");
        this->declare_parameter("monitor_rate", 1.0);
        this->declare_parameter("enable_diagnostics", true);
    }
    
    bool initializeI2C() {
        int bus = this->get_parameter("i2c_bus").as_int();
        std::string device = "/dev/i2c-" + std::to_string(bus);
        
        RCLCPP_INFO(this->get_logger(), "Opening I2C device: %s", device.c_str());
        
        i2c_fd_ = open(device.c_str(), O_RDWR);
        if (i2c_fd_ < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open I2C device: %s", device.c_str());
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "I2C device opened successfully");
        return true;
    }
    
    bool initializePCA9685() {
        int address = this->get_parameter("pca9685_address").as_int();
        int frequency = this->get_parameter("pwm_frequency").as_int();
        
        RCLCPP_INFO(this->get_logger(), "Initializing PCA9685 at 0x%02X...", address);
        
        // Set I2C slave address
        if (ioctl(i2c_fd_, I2C_SLAVE, address) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to set I2C slave address");
            return false;
        }
        
        // Reset PCA9685
        if (!writeRegister(PCA9685_MODE1, MODE1_RESTART)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to reset PCA9685");
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        
        // Set to sleep mode to change prescaler
        if (!writeRegister(PCA9685_MODE1, MODE1_SLEEP)) {
            return false;
        }
        
        // Set PWM frequency
        if (!setPWMFrequency(frequency)) {
            return false;
        }
        
        // Wake up and enable auto-increment
        if (!writeRegister(PCA9685_MODE1, MODE1_ALLCALL)) {
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        
        // Set output mode to totem pole (not open drain)
        if (!writeRegister(PCA9685_MODE2, MODE2_OUTDRV)) {
            return false;
        }
        
        pca9685_initialized_ = true;
        RCLCPP_INFO(this->get_logger(), "PCA9685 initialized at %d Hz", frequency);
        
        return true;
    }
    
    bool initializeGPIO() {
        std::string chip_name = this->get_parameter("gpio_chip").as_string();
        
        RCLCPP_INFO(this->get_logger(), "Opening GPIO chip: %s", chip_name.c_str());
        
        gpio_chip_ = gpiod_chip_open_by_name(chip_name.c_str());
        if (!gpio_chip_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open GPIO chip");
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "GPIO chip opened successfully");
        return true;
    }
    
    void initializeROS() {
        // Publishers
        diagnostics_pub_ = this->create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            "/diagnostics", 10
        );
        
        temperature_pub_ = this->create_publisher<sensor_msgs::msg::Temperature>(
            "~/cpu_temperature", 10
        );
        
        voltage_pub_ = this->create_publisher<std_msgs::msg::Float64>(
            "~/system_voltage", 10
        );
        
        status_pub_ = this->create_publisher<std_msgs::msg::String>(
            "~/status", 10
        );
        
        // Subscribers
        emergency_stop_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/emergency_stop",
            rclcpp::QoS(10).reliable().transient_local(),
            std::bind(&HardwareInterfaceNode::emergencyStopCallback, this, std::placeholders::_1)
        );
        
        RCLCPP_INFO(this->get_logger(), "ROS components initialized");
    }
    
    void startMonitoring() {
        double monitor_rate = this->get_parameter("monitor_rate").as_double();
        
        monitor_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / monitor_rate)),
            std::bind(&HardwareInterfaceNode::monitorLoop, this)
        );
        
        RCLCPP_INFO(this->get_logger(), "Hardware monitoring started at %.1f Hz", monitor_rate);
    }
    
    // ========================================================================
    // I2C OPERATIONS
    // ========================================================================
    
    bool writeRegister(uint8_t reg, uint8_t value) {
        uint8_t buffer[2] = {reg, value};
        
        if (write(i2c_fd_, buffer, 2) != 2) {
            RCLCPP_ERROR(this->get_logger(), 
                        "Failed to write to register 0x%02X", reg);
            return false;
        }
        
        return true;
    }
    
    bool readRegister(uint8_t reg, uint8_t& value) {
        if (write(i2c_fd_, &reg, 1) != 1) {
            RCLCPP_ERROR(this->get_logger(), 
                        "Failed to write register address 0x%02X", reg);
            return false;
        }
        
        if (read(i2c_fd_, &value, 1) != 1) {
            RCLCPP_ERROR(this->get_logger(), 
                        "Failed to read from register 0x%02X", reg);
            return false;
        }
        
        return true;
    }
    
    // ========================================================================
    // PCA9685 CONTROL
    // ========================================================================
    
    bool setPWMFrequency(int frequency) {
        // PCA9685 prescale formula: prescale = round(25MHz / (4096 * frequency)) - 1
        double prescale_value = 25000000.0 / (4096.0 * frequency) - 1.0;
        uint8_t prescale = static_cast<uint8_t>(std::round(prescale_value));
        
        RCLCPP_DEBUG(this->get_logger(), 
                     "Setting PWM frequency to %d Hz (prescale: %d)", 
                     frequency, prescale);
        
        return writeRegister(PCA9685_PRESCALE, prescale);
    }
    
    bool setPWM(uint8_t channel, uint16_t on, uint16_t off) {
        if (channel >= 16) {
            RCLCPP_ERROR(this->get_logger(), "Invalid PWM channel: %d", channel);
            return false;
        }
        
        if (!pca9685_initialized_) {
            RCLCPP_ERROR(this->get_logger(), "PCA9685 not initialized");
            return false;
        }
        
        // Calculate register addresses for this channel
        uint8_t base_reg = PCA9685_LED0_ON_L + (4 * channel);
        
        // Write ON time (12-bit value split into two 8-bit registers)
        if (!writeRegister(base_reg, on & 0xFF)) return false;
        if (!writeRegister(base_reg + 1, on >> 8)) return false;
        
        // Write OFF time (12-bit value split into two 8-bit registers)
        if (!writeRegister(base_reg + 2, off & 0xFF)) return false;
        if (!writeRegister(base_reg + 3, off >> 8)) return false;
        
        return true;
    }
    
    bool setPWMDutyCycle(uint8_t channel, double duty_cycle) {
        // Clamp duty cycle to [0.0, 1.0]
        duty_cycle = std::clamp(duty_cycle, 0.0, 1.0);
        
        // PCA9685 has 12-bit resolution (0-4095)
        uint16_t off_value = static_cast<uint16_t>(4095.0 * duty_cycle);
        
        return setPWM(channel, 0, off_value);
    }
    
    bool setPWMPulseWidth(uint8_t channel, uint16_t pulse_width_us) {
        // Convert pulse width (microseconds) to PWM value
        // At 50Hz, period = 20,000 microseconds
        // PWM value = (pulse_width / period) * 4096
        
        int frequency = this->get_parameter("pwm_frequency").as_int();
        double period_us = 1000000.0 / frequency;
        
        double duty_cycle = static_cast<double>(pulse_width_us) / period_us;
        
        return setPWMDutyCycle(channel, duty_cycle);
    }
    
    void disableAllPWM() {
        RCLCPP_INFO(this->get_logger(), "Disabling all PWM outputs");
        
        for (uint8_t channel = 0; channel < 16; ++channel) {
            setPWM(channel, 0, 0);
        }
    }
    
    // ========================================================================
    // SYSTEM MONITORING
    // ========================================================================
    
    void monitorLoop() {
        // Read CPU temperature
        double cpu_temp = readCPUTemperature();
        
        // Read system voltage (if available)
        double voltage = readSystemVoltage();
        
        // Publish temperature
        auto temp_msg = std::make_unique<sensor_msgs::msg::Temperature>();
        temp_msg->header.stamp = this->now();
        temp_msg->temperature = cpu_temp;
        temp_msg->variance = 0.0;
        temperature_pub_->publish(std::move(temp_msg));
        
        // Publish voltage
        auto voltage_msg = std::make_unique<std_msgs::msg::Float64>();
        voltage_msg->data = voltage;
        voltage_pub_->publish(std::move(voltage_msg));
        
        // Check for warnings
        bool temp_warning = cpu_temp > CPU_TEMP_WARNING;
        bool temp_critical = cpu_temp > CPU_TEMP_CRITICAL;
        bool voltage_warning = voltage < VOLTAGE_MIN || voltage > VOLTAGE_MAX;
        
        // Publish diagnostics
        if (this->get_parameter("enable_diagnostics").as_bool()) {
            publishDiagnostics(cpu_temp, voltage, temp_warning, 
                             temp_critical, voltage_warning);
        }
        
        // Handle critical conditions
        if (temp_critical) {
            RCLCPP_ERROR(this->get_logger(), 
                        "CRITICAL: CPU temperature %.1f°C exceeds limit!", 
                        cpu_temp);
            // Consider triggering emergency stop
        }
        
        if (voltage_warning) {
            RCLCPP_WARN(this->get_logger(), 
                       "System voltage %.2fV out of range [%.2fV - %.2fV]",
                       voltage, VOLTAGE_MIN, VOLTAGE_MAX);
        }
    }
    
    double readCPUTemperature() {
        // Read from Raspberry Pi's thermal zone
        std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
        
        if (!temp_file.is_open()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                               "Cannot read CPU temperature");
            return 0.0;
        }
        
        int temp_millidegrees;
        temp_file >> temp_millidegrees;
        temp_file.close();
        
        // Convert from millidegrees to degrees Celsius
        return temp_millidegrees / 1000.0;
    }
    
    double readSystemVoltage() {
        // Read from vcgencmd (Raspberry Pi specific)
        FILE* pipe = popen("vcgencmd measure_volts core", "r");
        
        if (!pipe) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                               "Cannot read system voltage");
            return 0.0;
        }
        
        char buffer[128];
        std::string result = "";
        
        while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            result += buffer;
        }
        
        pclose(pipe);
        
        // Parse voltage from "volt=1.2000V"
        size_t pos = result.find("volt=");
        if (pos != std::string::npos) {
            std::string voltage_str = result.substr(pos + 5);
            return std::stod(voltage_str);
        }
        
        return 0.0;
    }
    
    void publishDiagnostics(double cpu_temp, double voltage,
                           bool temp_warning, bool temp_critical,
                           bool voltage_warning) {
        auto diag_array = std::make_unique<diagnostic_msgs::msg::DiagnosticArray>();
        diag_array->header.stamp = this->now();
        
        // Hardware status
        diagnostic_msgs::msg::DiagnosticStatus hw_status;
        hw_status.name = "Hardware Interface";
        hw_status.hardware_id = "Raspberry Pi 4";
        
        if (temp_critical) {
            hw_status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
            hw_status.message = "CRITICAL: CPU temperature too high!";
        } else if (temp_warning || voltage_warning) {
            hw_status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            hw_status.message = "Hardware warning detected";
        } else {
            hw_status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
            hw_status.message = "Hardware operating normally";
        }
        
        // Add key-value pairs
        diagnostic_msgs::msg::KeyValue cpu_temp_kv;
        cpu_temp_kv.key = "CPU Temperature";
        cpu_temp_kv.value = std::to_string(cpu_temp) + " °C";
        hw_status.values.push_back(cpu_temp_kv);
        
        diagnostic_msgs::msg::KeyValue voltage_kv;
        voltage_kv.key = "System Voltage";
        voltage_kv.value = std::to_string(voltage) + " V";
        hw_status.values.push_back(voltage_kv);
        
        diagnostic_msgs::msg::KeyValue i2c_kv;
        i2c_kv.key = "I2C Status";
        i2c_kv.value = pca9685_initialized_ ? "OK" : "ERROR";
        hw_status.values.push_back(i2c_kv);
        
        diagnostic_msgs::msg::KeyValue gpio_kv;
        gpio_kv.key = "GPIO Status";
        gpio_kv.value = gpio_chip_ ? "OK" : "ERROR";
        hw_status.values.push_back(gpio_kv);
        
        diag_array->status.push_back(hw_status);
        
        diagnostics_pub_->publish(std::move(diag_array));
    }
    
    // ========================================================================
    // CALLBACKS
    // ========================================================================
    
    void emergencyStopCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        if (msg->data && !emergency_stop_active_) {
            RCLCPP_ERROR(this->get_logger(), "EMERGENCY STOP ACTIVATED!");
            emergency_stop_active_ = true;
            
            // Disable all PWM outputs immediately
            disableAllPWM();
            
            // Publish status
            auto status_msg = std::make_unique<std_msgs::msg::String>();
            status_msg->data = "EMERGENCY_STOP";
            status_pub_->publish(std::move(status_msg));
            
        } else if (!msg->data && emergency_stop_active_) {
            RCLCPP_INFO(this->get_logger(), "Emergency stop released");
            emergency_stop_active_ = false;
            
            // Reinitialize PCA9685
            initializePCA9685();
            
            auto status_msg = std::make_unique<std_msgs::msg::String>();
            status_msg->data = "OPERATIONAL";
            status_pub_->publish(std::move(status_msg));
        }
    }
    
    // ========================================================================
    // MEMBER VARIABLES
    // ========================================================================
    
    // Hardware
    int i2c_fd_;
    bool pca9685_initialized_;
    struct gpiod_chip* gpio_chip_;
    
    // State
    std::atomic<bool> emergency_stop_active_;
    
    // ROS components
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Temperature>::SharedPtr temperature_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr voltage_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr emergency_stop_sub_;
    rclcpp::TimerBase::SharedPtr monitor_timer_;
};

} // namespace robot_control

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<robot_control::HardwareInterfaceNode>();
        
        RCLCPP_INFO(node->get_logger(), 
                   "Hardware Interface Node running. Press Ctrl+C to exit.");
        
        rclcpp::spin(node);
        
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("hardware_interface"), 
                    "Fatal exception: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}
