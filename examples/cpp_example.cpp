/*
 * Shiva 2.0 C++ Example — Modern C++ RAII Integration
 *
 * Demonstrates C++17 RAII integration for ROS2 nodes, robotics software, and C++ flight stacks.
 *
 * Compilation command:
 *   g++ -std=c++17 -O3 cpp_example.cpp -I../bindings -L../target/release -lshiva -o cpp_example
 */

#include <iostream>
#include "../bindings/cpp/shiva.hpp"

int main() {
    std::cout << "=== Shiva 2.0 C++ RAII Control Loop ===" << std::endl;

    try {
        // 1. Initialize Shiva runtime using modern C++ RAII wrapper
        shiva::ShivaRuntime runtime(30, -1.0f, 1.0f);
        std::cout << "Shiva C++ runtime initialized successfully." << std::endl;

        // 2. Prepare sensor input struct
        auto input = shiva::ShivaRuntime::create_default_input();

        // 3. Execute 5-step control loop
        for (uint64_t step = 1; step <= 5; ++step) {
            input.timestep = step;
            for (size_t i = 0; i < 64; ++i) {
                input.state[i] = 0.1f * step;
            }

            shiva::OutputPacket output = runtime.step(input);

            std::cout << "Timestep " << step 
                      << " | Action[0]: " << output.final_action[0]
                      << " | Reward: " << output.reward 
                      << " | Emergency Veto: " << (output.mask[0] != 0 ? "YES" : "NO")
                      << std::endl;
        }

        std::cout << "=== Shiva C++ Loop Executed Cleanly ===" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "C++ Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
