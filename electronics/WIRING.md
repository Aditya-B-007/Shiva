# Bhujaha (भुजः) — Electronics & Wiring Integration Guide

This guide details the complete electrical architecture, pin mappings, and power distribution for the Bhujaha 3-DOF robot manipulator.

---

## 1. Electrical Architecture Overview

```
 [12V 10A SMPS Mains Power]
        │
        ├───▶ [High-Torque Serial Bus Servos (Joints 1, 2, 3)] (12V VCC Rail)
        │
        └───▶ [LM2596 / MP1584 Buck Converter (12V ➔ 5V 3A)]
                     │
                     ├───▶ [ESP32 / RP2040 Microcontroller (5V Vin)]
                     ├───▶ [MG90S Gripper Servo (5V Rail)]
                     └───▶ [Status LEDs & Push Buttons]
```

---

## 2. Microcontroller Pinout (ESP32 DevKit V1)

| ESP32 Pin | Connected Component | Function / Notes |
| :--- | :--- | :--- |
| **GPIO 16 (RX2)** | Serial Servo Driver RX | Half-duplex UART communication with STS3215 servos |
| **GPIO 17 (TX2)** | Serial Servo Driver TX | Half-duplex UART communication with STS3215 servos |
| **GPIO 18** | MG90S Gripper Servo PWM | 50 Hz PWM servo control signal |
| **GPIO 25** | "Teach / Record" Push Button | Pull-up input (pressed = LOW), toggles kinesthetic recording |
| **GPIO 26** | "Execute / Replay" Push Button| Pull-up input (pressed = LOW), triggers autonomous replay |
| **GPIO 27** | Emergency Stop Switch (NC) | Hardware interrupt for instant motor cutoff |
| **GPIO 32** | Status LED (Green) | Solid = Ready, Blinking = Replaying |
| **GPIO 33** | Record LED (Red) | Blinking = Recording user demonstration |
| **GND** | Common Ground | Common GND between ESP32, 12V SMPS, and Buck Converter |
| **5V (Vin)** | Buck Converter 5V Output | Logic power supply |

---

## 3. Half-Duplex Serial Servo Bus Interface

The Feetech STS3215 / SCS15 servos communicate over a single-wire half-duplex UART at 1,000,000 baud (1 Mbps).

### Option A: Dedicated Driver Board (Recommended)
Use a standard **Feetech URT-1** or **Waveshare Bus Servo Adapter board**:
* Connect `5V`, `GND`, `TX` (GPIO 17), `RX` (GPIO 16) from ESP32 to adapter.
* Connect external 12V power supply to the adapter's terminal block.
* Daisy-chain Joint 1, Joint 2, and Joint 3 using 3-pin JST cables.

### Option B: DIY 74HC126 / MOSFET Half-Duplex Circuit
* Join TX and RX through a bidirectional level-shifting buffer or 74HC126 tristate buffer controlled by a direction GPIO.

---

## 4. Grounding & Decoupling Best Practices
1. **Common Ground**: Ensure the 12V power supply ground, 5V buck converter ground, and ESP32 GND are bonded together.
2. **Bulk Capacitors**: Place a 1000 µF 25V electrolytic capacitor across the 12V power input near the first joint to absorb inductive back-EMF spikes during rapid deceleration.
3. **Emergency Stop**: The E-Stop button should physically break the 12V VCC line to the actuators (or trigger a relay) for fail-safe physical isolation.
