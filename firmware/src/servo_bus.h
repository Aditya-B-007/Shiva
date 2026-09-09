#pragma once
#include <Arduino.h>
#include <stdint.h>

// Driver for Feetech STS3215 / SCS15 High-Torque Serial Bus Servos
class ServoBus {
public:
    ServoBus(HardwareSerial &serial, int rxPin, int txPin, long baud = 1000000);

    void begin();
    
    // Joint angle queries (in radians, converted from 12-bit 0-4095 range)
    bool readPosition(uint8_t id, float &out_radians);
    bool readCurrent(uint8_t id, float &out_current_amps);

    // Motor control commands
    bool writePosition(uint8_t id, float target_radians, uint16_t time_ms = 0, uint16_t speed = 0);
    bool setTorqueEnable(uint8_t id, bool enable);
    bool setAllTorque(bool enable);

private:
    HardwareSerial &busSerial;
    int _rxPin, _txPin;
    long _baud;

    // Protocol constants
    static const uint8_t INST_READ = 0x02;
    static const uint8_t INST_WRITE = 0x03;
    static const uint8_t REG_TORQUE_ENABLE = 0x28;
    static const uint8_t REG_PRESENT_POSITION = 0x38;
    static const uint8_t REG_PRESENT_CURRENT = 0x45;
    static const uint8_t REG_GOAL_POSITION = 0x2A;

    uint8_t calcChecksum(const uint8_t *buffer, size_t length);
    bool sendPacket(uint8_t id, uint8_t instruction, const uint8_t *params, size_t param_len);
    bool readResponse(uint8_t id, uint8_t *out_data, size_t expected_len);
};
