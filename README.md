# Bhujaha (भुजः)

### Open-Source 3-DOF Household Robotic Manipulator Powered by the Shiva Real-Time Cognitive Framework

> **Mission**: Democratize household robotics by delivering a fully open-source, 3-DOF robotic arm that anyone can manufacture for **under ₹20,000 INR (~$240 USD)**. Built with a **"Teach-Once, Replay Forever"** paradigm, empowering everyday users to automate repetitive physical chores safely and effortlessly.

---

## 1. Overview

Industrial robotic manipulators often cost tens of thousands of dollars, require specialized programming expertise, and are locked behind closed proprietary ecosystems. **Bhujaha (भुजः)** flips this paradigm on its head.

Bhujaha is a low-cost, open-source 3-DOF robotic manipulator engineered for domestic environments. By combining commodity 3D-printed structural components, accessible off-the-shelf actuators, and the deterministic, sub-millisecond cognitive intelligence of **Shiva**, Bhujaha makes physical automation practical, safe, and accessible to any household.

---

## 2. Core Philosophy: Teach-Once, Replay Forever

Users should not need a robotics degree or ROS expertise to automate tasks in their home. Bhujaha introduces intuitive **Kinesthetic Lead-Through Teaching**:

```
[ User Physically Demonstrates Task ]
                  │
                  ▼
[ 12-bit Magnetic Encoders Stream Joint Angles (50-100 Hz) ]
                  │
                  ▼
[ Shiva Kinematic Ingestion & Trajectory Spline Smoothing ]
                  │
                  ▼
[ Shiva CPO Safety GuardRails & Workspace Boundary Checks ]
                  │
                  ▼
[ Autonomous Execution & Continuous Task Replay ]
```

1. **Lead-Through Demonstration**: The user presses a physical "Record / Teach" button. The joint actuators enter a compliant, zero-torque back-drivable state. The user simply guides the arm through the target task by hand (e.g., wiping a countertop, watering a plant, or picking and placing an object).
2. **Signal Conditioning & Trajectory Processing**: Shiva ingests the joint angles ($\theta_1, \theta_2, \theta_3$) in real time. It filters human tremor and jerk using cubic/quintic splines, yielding mathematically continuous and smooth trajectories.
3. **Deterministic Safety Projection**: The conditioned trajectory is evaluated against Shiva’s safety guardrails. Shiva enforces speed clamps, acceleration boundaries, torque limitations, and geometric virtual walls (preventing the robot from colliding with tables or surrounding obstacles).
4. **Autonomous Execution**: The user presses "Execute", and Bhujaha deterministically repeats the demonstrated task indefinitely with closed-loop precision.

---

## 3. How Shiva Powers Bhujaha

The brain behind Bhujaha is the **Shiva Framework** (`shivaFramework`), a sub-millisecond autonomous runtime written in zero-allocation Rust. 

### Why Shiva is Essential for Household Arms:
* **Microsecond-Level Determinism**: Control loops evaluate in microseconds, enabling fast reaction times to unexpected external contacts, slips, or stalls.
* **Guaranteed Physical Safety (CPO GuardRails)**: In domestic settings around family members, children, and pets, safety cannot be compromised. Shiva's Constrained Policy Optimization (CPO) guardrails project actuator setpoints onto strictly verified convex safety sets before sending commands to physical motors.
* **Sensor-to-Actuator Sliding Window**: Shiva’s `EnvironmentMatrix` retains a sliding historical window of joint positions, velocities, and motor currents, enabling early anomaly detection (such as catching a jammed gripper or a physical collision).
* **Multi-Node Consensus Architecture**: Coordinates trajectory tracking, real-time risk evaluation, and hardware safety limits into a synchronized control pipeline.

---

## 4. Household Applications: Betterment in Daily Living

Bhujaha is designed to assist humans with mundane, repetitive, or physically taxing tasks:

### Domestic Chore Automation
* **Countertop & Table Wiping**: Moving a sponge or microfiber pad across flat surfaces in consistent raster patterns.
* **Sorting & Pick-and-Place**: Sorting laundry pegs, organizing cutlery, or moving items between bins and tabletops.
* **Plant Care**: Delivering measured amounts of water to indoor potted plants on scheduled intervals.
* **Kitchen Assistance**: Repetitive stirring, holding utensils, or moving containers between preparation stations.

### Accessibility & Assisted Living
* **Assistance for the Elderly & Disabled**: Holding objects steadily, retrieving dropped items within reach, or repositioning bedside items.
* **Strain Reduction**: Eliminating repetitive strain for individuals with arthritis or limited upper-body mobility.

### Privacy-First Domestic Robotics
* **100% On-Device / Zero Cloud**: Bhujaha runs entirely on local edge hardware. No video feeds, sensor telemetry, or voice data ever leave the home.
* **No Subscriptions**: Complete open-source ownership of both hardware designs and software runtime.

---

## 5. Bill of Materials (BOM) Target (< ₹20,000 INR)

Bhujaha is designed from the ground up to respect a strict manufacturing ceiling of ₹20,000 INR (~$240 USD):

| Component | Recommended Hardware | Purpose | Estimated Cost (INR) |
| :--- | :--- | :--- | :--- |
| **Actuators (Joints 1, 2, 3)** | 3× High-Torque Serial Bus Servos (e.g. STS3215 / SCS15) | Back-drivable, high torque, 12-bit magnetic encoder feedback over UART bus | ₹7,500 – ₹8,500 |
| **End Effector / Gripper** | 1× Micro Servo (MG90S) + 3D printed mechanical jaw | Pick, grasp, and hold domestic objects | ₹300 – ₹500 |
| **Chassis & Arm Links** | 3D Printed PETG / PLA+ structural parts + 2020 Aluminum extrusions | Lightweight, rigid, easily replaceable chassis | ₹2,500 – ₹3,500 |
| **Bearings & Fasteners** | 608RS bearings, thrust bearings, M3/M4/M5 hardware | Smooth joint articulation and backlash reduction | ₹1,000 – ₹1,500 |
| **Microcontroller Bridge** | ESP32 or RP2040 (Raspberry Pi Pico) | High-speed encoder reading and motor command streaming | ₹1,000 – ₹1,500 |
| **Power Supply** | 12V 10A DC SMPS | Stable power supply for all joint actuators | ₹1,200 – ₹1,800 |
| **Control UI & Peripherals** | Record/Play physical push buttons, emergency stop switch, silicon wiring | Direct tactile operation without needing a computer screen | ₹1,000 – ₹1,500 |
| **Total Estimated Hardware Cost** | | | **~₹14,500 – ₹18,500** |

---

## 6. Repository Structure

```
.
├── Cargo.toml               # Workspace manifest
├── README.md                # Project documentation & overview
├── openscad/                # Parametric 3D CAD models (OpenSCAD)
│   ├── parameters.scad      # Global dimensions & hardware specs
│   ├── base.scad            # Turntable & pedestal (Joint 1)
│   ├── shoulder_link.scad   # Upper arm link (Joint 2)
│   ├── elbow_link.scad      # Forearm link (Joint 3)
│   ├── gripper.scad         # Micro parallel-jaw gripper
│   └── bhujaha_assembly.scad# Kinematic assembly with joint angles
├── electronics/             # Electrical integration & hardware docs
│   ├── BOM.md               # Bill of materials target (< ₹20,000 INR)
│   ├── WIRING.md            # Pinouts, power rails & bus wiring
│   └── schematic.ascii      # Complete wiring schematic
├── firmware/                # Microcontroller software (ESP32 / RP2040)
│   ├── platformio.ini       # PlatformIO build configuration
│   └── src/
│       ├── main.cpp         # 50 Hz control loop & button handler
│       ├── servo_bus.h/.cpp # STS3215 serial bus servo driver
│       ├── trajectory_buffer.h # On-device teach trajectory storage
│       └── shiva_packet.h   # Binary protocol with Shiva runtime
└── shivaFramework/          # Core Shiva real-time cognitive runtime
    ├── Cargo.toml           # Shiva framework package
    └── src/
        ├── lib.rs           # Framework root & prelude
        ├── runtime.rs       # ShivaRuntime lifecycle orchestrator
        ├── config.rs        # ShivaBuilder & configuration engine
        ├── framework/       # Core traits, safety contracts & error types
        ├── nodes/           # 5-Node Mothership Ensemble & pipeline
        ├── brain/           # Middleware facades & DTOs
        ├── algorithms/      # Mathematical models (SAC, CPO, TD3, IQN, RND)
        ├── environment/     # Sliding window matrix & actuator signals
        ├── protocol/        # System input/output data transfer objects
        └── adapters/        # Hardware and input mapping adapters
```

---

## 7. Getting Started

### Prerequisites
* [Rust toolchain](https://rustup.rs/) (edition 2021 or later).

### Building the Shiva Runtime
Clone the repository and compile the workspace:

```bash
# Clone the repository
git clone https://github.com/Aditya-B-007/Shiva.git
cd Shiva

# Build the framework
cargo build --release

# Run unit tests and verification suite
cargo test
```

---

## 8. License

Distributed under the **Apache License 2.0**. See `LICENSE` for details.
