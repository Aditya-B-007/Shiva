// Shiva 2.0 — Crate Root & Framework API
//
// WHAT THIS FILE DOES:
// Registers all 4 architectural layers of the Shiva runtime + framework configuration
// and exposes a clean `prelude` module for external developers using the framework.
// Also registers `ffi` module for C, C++, and Python cross-language integration.
//
// MODULE STRUCTURE:
// ├── algorithms  — Layer 1: Pure RL mathematics (SAC, CPO, IQN, TD3+z, RND)
// ├── brain       — Layer 2: Decoupling middleware & SIMD DTOs
// ├── environment — State window queue (EnvironmentMatrix) & actuator buffer (ActuatorSignal)
// ├── nodes       — Layer 3: 5-Node Mothership Ensemble & Orchestrator
// ├── protocol    — Protocol communication (systemSide, shivaSide, middleMan)
// ├── config      — Framework configuration engine (ShivaConfig, ShivaBuilder)
// └── ffi         — C-ABI exports for C, C++, Python, ROS2, and RTOS bindings

pub mod algorithms;
pub mod brain;
pub mod config;
pub mod environment;
pub mod ffi;
pub mod nodes;
pub mod protocol;

pub use protocol as cscp;

/// Framework Prelude exposing essential types for end-user applications
pub mod prelude {
    pub use crate::config::{ShivaBuilder, ShivaConfig};
    pub use crate::environment::{ActuatorSignal, EnvironmentMatrix, MatrixRow, NodeType};
    pub use crate::nodes::{EnvironmentStack, MothershipOrchestrator};
    pub use crate::protocol::{ManInTheMiddle, ShivaOutputDTO, SystemInputDTO};
}
