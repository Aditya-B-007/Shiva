// Shiva 2.0 — Framework Module Root
//
// WHAT THIS MODULE DOES:
// Defines the core framework contracts that constitute Shiva's public API.
// These are the stable interfaces that external developers code against
// to extend, customize, and integrate with the Shiva framework.
//
// HOW IT IS ORGANIZED:
// ├── error       — Structured error hierarchy (ShivaError, ConfigError, etc.)
// ├── dimensions  — Type aliases for state/action dimensions (StateVector, ActionVector)
// ├── lifecycle   — Lifecycle state machine (Created → Configured → ... → ShutDown)
// ├── node        — Node trait (universal pipeline extension point)
// ├── orchestrator — Orchestrator trait (pluggable pipeline coordinator)
// ├── safety      — SafetyPolicy trait (mandatory safety contract)
// ├── context     — ExecutionContext (per-cycle data envelope)
// ├── adapter     — InputAdapter / OutputAdapter traits (hardware abstraction)
// └── diagnostics — CycleDiagnostics (runtime observability)
//
// WHY WE DO THIS:
// The framework module separates stable public contracts from implementation
// details. External developers depend only on these traits and types.
// Internal module restructuring (e.g., changing algorithm implementations)
// does not break framework consumers.

pub mod error;
pub mod dimensions;
pub mod lifecycle;
pub mod node;
pub mod orchestrator;
pub mod safety;
pub mod context;
pub mod adapter;
pub mod diagnostics;

// ═══════════════════════════════════════════════════════════════
// Re-exports: Framework contracts available as `framework::*`
// ═══════════════════════════════════════════════════════════════

// Error hierarchy
pub use error::{ShivaError, ConfigError, SafetyError, TransportError, RuntimeError};

// Dimension types
pub use dimensions::{
    StateVector, ActionVector, RuleFlagVector, SkillIdVector,
    DEFAULT_STATE_DIM, DEFAULT_ACTION_DIM, DEFAULT_SKILL_DIM,
};

// Lifecycle
pub use lifecycle::{LifecycleState, LifecycleAware};

// Node interface
pub use node::{Node, Phase, NodeOutcome};

// Orchestrator interface
pub use orchestrator::Orchestrator;

// Safety interface
pub use safety::{SafetyPolicy, SafetyVerdict};

// Execution context
pub use context::ExecutionContext;

// Adapters
pub use adapter::{InputAdapter, OutputAdapter, AdaptedInput, DispatchResult};

// Diagnostics
pub use diagnostics::{CycleDiagnostics, ConsensusWeights};
