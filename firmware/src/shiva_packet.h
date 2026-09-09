#pragma once
#include <stdint.h>

// Shiva 2.0 Physical Telemetry & Command Protocol
// Packets exchanged between Microcontroller (ESP32/Pico) and Shiva Runtime

#define SHIVA_MAGIC_HEADER 0x5348 // "SH" in hex

#pragma pack(push, 1)

// Telemetry packet sent from arm electronics to Shiva (Joint state + Currents)
struct TelemetryPacket {
    uint16_t header;         // Always SHIVA_MAGIC_HEADER
    uint32_t timestamp_ms;   // Local millisecond timestamp
    float theta[3];          // Joint angles (rad)
    float current[3];        // Joint currents (A / raw)
    float gripper_pos;       // Gripper aperture (0.0 = closed, 1.0 = open)
    uint8_t system_state;    // 0 = IDLE, 1 = TEACHING, 2 = REPLAYING, 3 = ESTOP
    uint8_t checksum;        // XOR checksum of packet bytes
};

// Command packet received from Shiva into arm electronics (Actuator setpoints)
struct CommandPacket {
    uint16_t header;         // Always SHIVA_MAGIC_HEADER
    float target_theta[3];   // Target joint angles (rad)
    float target_gripper;    // Target gripper aperture (0.0 to 1.0)
    uint16_t execution_time; // Time to reach target in ms
    uint8_t flags;           // Bit 0 = Torque enable, Bit 1 = E-stop
    uint8_t checksum;        // XOR checksum
};

#pragma pack(pop)
