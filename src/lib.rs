// Shiva 2.0 — Crate Root & Framework API
//
// WHAT THIS FILE DOES:
// Registers all architectural layers of the Shiva runtime + framework contracts
// and exposes a clean `prelude` module for external developers using the framework.
// Also registers `ffi` module for C, C++, and Python cross-language integration.
//
// MODULE STRUCTURE:
// ├── framework   — Framework contracts (Node, Orchestrator, SafetyPolicy, etc.)
// ├── algorithms  — Layer 1: Pure RL mathematics (SAC, CPO, IQN, TD3+z, RND)
// ├── brain       — Layer 2: Decoupling middleware & SIMD DTOs
// ├── environment — State window queue (EnvironmentMatrix) & actuator buffer (ActuatorSignal)
// ├── nodes       — Layer 3: 5-Node Mothership Ensemble & Orchestrator
// ├── protocol    — Protocol communication (systemSide, shivaSide, middleMan)
// ├── adapters    — Concrete InputAdapter / OutputAdapter implementations
// ├── config      — Framework configuration engine (ShivaConfig, ShivaBuilder)
// ├── runtime     — ShivaRuntime (primary framework entry point)
// └── ffi         — C-ABI exports for C, C++, Python, ROS2, and RTOS bindings

pub mod framework;
pub mod algorithms;
pub mod brain;
pub mod config;
pub mod environment;
pub mod adapters;
pub mod ffi;
pub mod nodes;
pub mod protocol;
pub mod runtime;

pub use protocol as cscp;

/// Stable Public API — Framework users import from here.
///
/// Types and traits in this module are part of the public stability guarantee.
/// Internal restructuring should not break code that imports only from `prelude`.
pub mod prelude {
    // ═══════════════════════════════════════════════════════════════
    // Framework Contracts (stable)
    // ═══════════════════════════════════════════════════════════════
    pub use crate::framework::{
        // Error hierarchy
        ShivaError, ConfigError, SafetyError, TransportError, RuntimeError,
        // Node interface
        Node, Phase, NodeOutcome,
        // Orchestrator interface
        Orchestrator,
        // Safety interface
        SafetyPolicy, SafetyVerdict,
        // Execution context
        ExecutionContext,
        // Adapters
        InputAdapter, OutputAdapter, AdaptedInput, DispatchResult,
        // Lifecycle
        LifecycleState, LifecycleAware,
        // Dimensions
        StateVector, ActionVector, RuleFlagVector, SkillIdVector,
        // Diagnostics
        CycleDiagnostics, ConsensusWeights,
    };

    // ═══════════════════════════════════════════════════════════════
    // Builder and Runtime (stable)
    // ═══════════════════════════════════════════════════════════════
    pub use crate::config::{ShivaBuilder, ShivaConfig};
    pub use crate::runtime::ShivaRuntime;

    // ═══════════════════════════════════════════════════════════════
    // DTOs (stable — C ABI)
    // ═══════════════════════════════════════════════════════════════
    pub use crate::environment::{ActuatorSignal, EnvironmentMatrix, MatrixRow, NodeType};
    pub use crate::nodes::{EnvironmentStack, MothershipOrchestrator};
    pub use crate::protocol::{ManInTheMiddle, ShivaOutputDTO, SystemInputDTO};
}

/// Internal implementation details — not part of the stability guarantee.
///
/// These modules are re-exported for advanced users who need direct access
/// to algorithm internals or brain-layer adapters. Breaking changes in these
/// modules may occur between minor versions.
#[doc(hidden)]
pub mod internals {
    pub use crate::algorithms;
    pub use crate::brain;
}
