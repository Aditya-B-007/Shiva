# Bhujaha (भुजः) — Embedded Microcontroller Firmware

This directory contains the real-time embedded firmware running directly on the robot's microcontroller (ESP32 DevKit V1 or RP2040 Pico).

## Responsibilities
1. **Actuator Communication**: Communicates with Feetech STS3215/SCS15 serial bus servos over 1 Mbps half-duplex UART to query 12-bit magnetic encoders and send position commands.
2. **Kinesthetic Teaching & Replay**: Manages the compliant "Teach" mode (torque disabled) and autonomous "Play" mode via physical push-button triggers.
3. **Emergency Stop (E-Stop)**: Hardware interrupt that instantly cuts off motor torque if the safety switch is tripped.
4. **Shiva Bridge Protocol**: Streams high-frequency joint telemetry packets (`TelemetryPacket`) over USB-UART to the host Shiva runtime, and receives safety-filtered actuator setpoints (`CommandPacket`).

## Directory Structure
* `src/main.cpp`: Main control loop, 50 Hz timing engine, button debounce, state machine, and serial telemetry pipeline.
* `src/servo_bus.h` / `src/servo_bus.cpp`: Hardware driver for serial bus servos.
* `src/trajectory_buffer.h`: On-device memory ring buffer for lead-through trajectory capture.
* `src/shiva_packet.h`: Packed binary protocol struct definitions matching Shiva's data transfer layer.
* `platformio.ini`: PlatformIO build configurations for ESP32 and RP2040.

## Flashing Instructions

### Using PlatformIO (Recommended)
```bash
cd firmware
pio run --target upload
```

### Using Arduino IDE
1. Open Arduino IDE.
2. Install the **ESP32 by Espressif Systems** board package via Board Manager.
3. Select board: **ESP32 Dev Module**.
4. Open `firmware/src/main.cpp`.
5. Connect your ESP32 via USB and click **Upload**.
