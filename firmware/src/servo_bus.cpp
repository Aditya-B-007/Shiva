#include "servo_bus.h"

ServoBus::ServoBus(HardwareSerial &serial, int rxPin, int txPin, long baud)
    : busSerial(serial), _rxPin(rxPin), _txPin(txPin), _baud(baud) {}

void ServoBus::begin() {
    busSerial.begin(_baud, SERIAL_8N1, _rxPin, _txPin);
}

uint8_t ServoBus::calcChecksum(const uint8_t *buffer, size_t length) {
    uint32_t sum = 0;
    for (size_t i = 2; i < length; i++) {
        sum += buffer[i];
    }
    return ~(sum & 0xFF);
}

bool ServoBus::sendPacket(uint8_t id, uint8_t instruction, const uint8_t *params, size_t param_len) {
    uint8_t packet[32];
    packet[0] = 0xFF;
    packet[1] = 0xFF;
    packet[2] = id;
    packet[3] = param_len + 2; // Length: instruction + params + checksum
    packet[4] = instruction;

    for (size_t i = 0; i < param_len; i++) {
        packet[5 + i] = params[i];
    }

    packet[5 + param_len] = calcChecksum(packet, 5 + param_len);

    while (busSerial.available()) busSerial.read(); // Flush input buffer
    busSerial.write(packet, 6 + param_len);
    busSerial.flush();
    return true;
}

bool ServoBus::readResponse(uint8_t id, uint8_t *out_data, size_t expected_len) {
    unsigned long start = millis();
    size_t count = 0;
    uint8_t resp[32];

    while (millis() - start < 15) { // 15 ms timeout
        if (busSerial.available()) {
            resp[count++] = busSerial.read();
            if (count >= 6 + expected_len) break;
        }
    }

    if (count < 6 + expected_len) return false;
    if (resp[0] != 0xFF || resp[1] != 0xFF || resp[2] != id) return false;

    for (size_t i = 0; i < expected_len; i++) {
        out_data[i] = resp[5 + i];
    }
    return true;
}

bool ServoBus::readPosition(uint8_t id, float &out_radians) {
    uint8_t params[2] = { REG_PRESENT_POSITION, 0x02 }; // Read 2 bytes
    if (!sendPacket(id, INST_READ, params, 2)) return false;

    uint8_t data[2];
    if (!readResponse(id, data, 2)) return false;

    int16_t raw_pos = (int16_t)(data[0] | (data[1] << 8));
    // STS3215 maps 0 to 4095 across 360 degrees (0 to 2*pi rad)
    out_radians = ((float)raw_pos / 4096.0f) * 2.0f * PI - PI;
    return true;
}

bool ServoBus::readCurrent(uint8_t id, float &out_current_amps) {
    uint8_t params[2] = { REG_PRESENT_CURRENT, 0x02 };
    if (!sendPacket(id, INST_READ, params, 2)) return false;

    uint8_t data[2];
    if (!readResponse(id, data, 2)) return false;

    int16_t raw_current = (int16_t)(data[0] | (data[1] << 8));
    out_current_amps = (float)raw_current * 0.0065f; // ~6.5 mA per LSB
    return true;
}

bool ServoBus::writePosition(uint8_t id, float target_radians, uint16_t time_ms, uint16_t speed) {
    // Convert radians (-pi to +pi) to 0 - 4095
    float norm = (target_radians + PI) / (2.0f * PI);
    if (norm < 0.0f) norm = 0.0f;
    if (norm > 1.0f) norm = 1.0f;
    uint16_t pos = (uint16_t)(norm * 4095.0f);

    uint8_t params[7];
    params[0] = REG_GOAL_POSITION;
    params[1] = pos & 0xFF;
    params[2] = (pos >> 8) & 0xFF;
    params[3] = time_ms & 0xFF;
    params[4] = (time_ms >> 8) & 0xFF;
    params[5] = speed & 0xFF;
    params[6] = (speed >> 8) & 0xFF;

    return sendPacket(id, INST_WRITE, params, 7);
}

bool ServoBus::setTorqueEnable(uint8_t id, bool enable) {
    uint8_t params[2] = { REG_TORQUE_ENABLE, (uint8_t)(enable ? 1 : 0) };
    return sendPacket(id, INST_WRITE, params, 2);
}

bool ServoBus::setAllTorque(bool enable) {
    bool ok = true;
    for (uint8_t id = 1; id <= 3; id++) {
        ok &= setTorqueEnable(id, enable);
        delayMicroseconds(500);
    }
    return ok;
}
