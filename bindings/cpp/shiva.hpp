/*
 * Shiva 2.0 — C++ Header-Only Wrapper
 *
 * Modern C++17 RAII Wrapper around the Shiva 2.0 C-ABI for seamless integration
 * into C++ robotics, ROS2 nodes, flight software, and simulation environments.
 */

#ifndef SHIVA_HPP
#define SHIVA_HPP

#include "../c/shiva.h"
#include <stdexcept>
#include <vector>
#include <array>
#include <memory>

namespace shiva {

/// C++ Wrapper for SystemInputDTO
using InputPacket = SystemInputDTO;

/// C++ Wrapper for ShivaOutputDTO
using OutputPacket = ShivaOutputDTO;

/**
 * Modern C++ RAII class managing a Shiva 2.0 autonomous control runtime engine.
 */
class ShivaRuntime {
public:
    /**
     * Constructs a Shiva 2.0 control runtime.
     *
     * @param matrix_rows Max sliding window matrix capacity (default: 20)
     * @param min_signal Minimum actuator limit (default: -1.0)
     * @param max_signal Maximum actuator limit (default: 1.0)
     */
    explicit ShivaRuntime(size_t matrix_rows = 20, float min_signal = -1.0f, float max_signal = 1.0f)
        : handle_(shiva_create(matrix_rows, min_signal, max_signal)) {
        if (!handle_) {
            throw std::runtime_error("Failed to initialize Shiva 2.0 runtime engine.");
        }
    }

    ~ShivaRuntime() {
        if (handle_) {
            shiva_destroy(handle_);
            handle_ = nullptr;
        }
    }

    // Prevent copying to maintain strict RAII handle ownership
    ShivaRuntime(const ShivaRuntime&) = delete;
    ShivaRuntime& operator=(const ShivaRuntime&) = delete;

    // Allow move semantics
    ShivaRuntime(ShivaRuntime&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    ShivaRuntime& operator=(ShivaRuntime&& other) noexcept {
        if (this != &other) {
            if (handle_) shiva_destroy(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /**
     * Creates a pre-populated SystemInputDTO struct initialized with safe defaults.
     */
    static InputPacket create_default_input() {
        InputPacket input;
        shiva_default_input(&input);
        return input;
    }

    /**
     * Executes a single 3-phase consensus control cycle (< 1 ms).
     *
     * @param input Sensor telemetry packet
     * @return Output control packet containing final safe actuator commands
     */
    OutputPacket step(const InputPacket& input) {
        OutputPacket output{};
        int32_t res = shiva_step(handle_, &input, &output);
        if (res != 0) {
            throw std::runtime_error("Shiva runtime step error code: " + std::to_string(res));
        }
        return output;
    }

private:
    ShivaHandle handle_{nullptr};
};

} // namespace shiva

#endif /* SHIVA_HPP */
