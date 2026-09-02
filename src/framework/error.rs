// Shiva 2.0 — Framework Error Model
//
// WHAT THIS FILE DOES:
// Defines the structured error hierarchy for the Shiva framework.
// All framework operations return errors from this hierarchy, enabling
// applications to programmatically distinguish failure modes.
//
// HOW IT DOES IT:
// Uses a top-level `ShivaError` enum with domain-specific sub-enums for
// configuration, node execution, safety, transport, and runtime errors.
//
// WHY WE DO THIS:
// A safety-critical control framework must propagate errors consistently.
// Internal failures should never be silently discarded. Structured errors
// allow applications to implement domain-appropriate recovery strategies.

use std::fmt;

/// Top-level framework error type.
///
/// WHAT: Classifies all possible framework failure modes.
/// HOW: Each variant wraps a domain-specific error sub-type.
/// WHY: Enables pattern-matching for domain-appropriate error handling.
#[derive(Debug)]
pub enum ShivaError {
    /// Configuration-time errors (invalid limits, NaN, bad dimensions).
    Configuration(ConfigError),

    /// Runtime errors from individual node execution.
    Node {
        /// Name of the node that failed.
        node_name: String,
        /// Underlying error from the node.
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Safety pipeline errors (veto failures, constraint violations).
    Safety(SafetyError),

    /// Hardware transport / actuator dispatch errors.
    Transport(TransportError),

    /// General runtime errors (lifecycle violations, internal invariants).
    Runtime(RuntimeError),
}

impl fmt::Display for ShivaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShivaError::Configuration(e) => write!(f, "Configuration error: {}", e),
            ShivaError::Node { node_name, source } => {
                write!(f, "Node '{}' error: {}", node_name, source)
            }
            ShivaError::Safety(e) => write!(f, "Safety error: {}", e),
            ShivaError::Transport(e) => write!(f, "Transport error: {}", e),
            ShivaError::Runtime(e) => write!(f, "Runtime error: {}", e),
        }
    }
}

impl std::error::Error for ShivaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ShivaError::Node { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Configuration Errors
// ═══════════════════════════════════════════════════════════════

/// Errors detected during framework configuration / construction.
///
/// WHAT: Captures invalid builder parameters before runtime begins.
/// WHY: Invalid configuration should fail at construction, not during execution.
#[derive(Debug)]
pub enum ConfigError {
    /// Actuator signal limits are invalid (e.g., min >= max).
    InvalidLimits {
        min: f32,
        max: f32,
        reason: &'static str,
    },

    /// A dimensional parameter is out of valid range.
    InvalidDimension {
        name: &'static str,
        value: usize,
    },

    /// A floating-point configuration value is NaN or infinite.
    NaNOrInfinite {
        field: &'static str,
    },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::InvalidLimits { min, max, reason } => {
                write!(f, "Invalid limits [{}, {}]: {}", min, max, reason)
            }
            ConfigError::InvalidDimension { name, value } => {
                write!(f, "Invalid dimension '{}': {}", name, value)
            }
            ConfigError::NaNOrInfinite { field } => {
                write!(f, "NaN or Infinite value in field '{}'", field)
            }
        }
    }
}

impl std::error::Error for ConfigError {}

// ═══════════════════════════════════════════════════════════════
// Safety Errors
// ═══════════════════════════════════════════════════════════════

/// Errors from the safety pipeline.
///
/// WHAT: Captures safety-layer failures and constraint violations.
/// WHY: Safety failures must be distinguishable from other error types
///      so downstream systems can take appropriate protective action.
#[derive(Debug)]
pub enum SafetyError {
    /// The safety policy vetoed the proposed action.
    ActionVetoed {
        reason: String,
    },

    /// The safety evaluator encountered an internal failure.
    EvaluationFailed {
        reason: String,
    },

    /// The safety pipeline was not properly configured.
    NotConfigured,
}

impl fmt::Display for SafetyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SafetyError::ActionVetoed { reason } => write!(f, "Action vetoed: {}", reason),
            SafetyError::EvaluationFailed { reason } => {
                write!(f, "Safety evaluation failed: {}", reason)
            }
            SafetyError::NotConfigured => write!(f, "Safety pipeline not configured"),
        }
    }
}

impl std::error::Error for SafetyError {}

// ═══════════════════════════════════════════════════════════════
// Transport Errors
// ═══════════════════════════════════════════════════════════════

/// Errors from hardware transport / actuator dispatch.
///
/// WHAT: Captures failures in the output dispatch pipeline.
/// WHY: Transport failures (CAN bus timeout, serial error, etc.) must be
///      propagated to the application layer, not silently discarded.
#[derive(Debug)]
pub enum TransportError {
    /// The dispatch operation failed.
    DispatchFailed(String),

    /// No staged signal was available for dispatch.
    NoStagedSignal,

    /// The transport layer is not connected.
    NotConnected,
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransportError::DispatchFailed(msg) => write!(f, "Dispatch failed: {}", msg),
            TransportError::NoStagedSignal => write!(f, "No staged signal to dispatch"),
            TransportError::NotConnected => write!(f, "Transport not connected"),
        }
    }
}

impl std::error::Error for TransportError {}

// ═══════════════════════════════════════════════════════════════
// Runtime Errors
// ═══════════════════════════════════════════════════════════════

/// Errors from the framework runtime.
///
/// WHAT: Captures lifecycle violations, internal invariant failures, etc.
/// WHY: Runtime errors indicate programming mistakes or environmental
///      conditions that prevent the framework from operating correctly.
#[derive(Debug)]
pub enum RuntimeError {
    /// An operation was attempted in an invalid lifecycle state.
    InvalidLifecycleTransition {
        from: String,
        to: String,
    },

    /// An internal invariant was violated.
    InvariantViolation(String),

    /// The orchestrator failed to produce a result.
    OrchestratorFailed(String),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeError::InvalidLifecycleTransition { from, to } => {
                write!(f, "Invalid lifecycle transition: {} -> {}", from, to)
            }
            RuntimeError::InvariantViolation(msg) => {
                write!(f, "Invariant violation: {}", msg)
            }
            RuntimeError::OrchestratorFailed(msg) => {
                write!(f, "Orchestrator failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for RuntimeError {}

// ═══════════════════════════════════════════════════════════════
// Convenience conversions
// ═══════════════════════════════════════════════════════════════

impl From<ConfigError> for ShivaError {
    fn from(e: ConfigError) -> Self {
        ShivaError::Configuration(e)
    }
}

impl From<SafetyError> for ShivaError {
    fn from(e: SafetyError) -> Self {
        ShivaError::Safety(e)
    }
}

impl From<TransportError> for ShivaError {
    fn from(e: TransportError) -> Self {
        ShivaError::Transport(e)
    }
}

impl From<RuntimeError> for ShivaError {
    fn from(e: RuntimeError) -> Self {
        ShivaError::Runtime(e)
    }
}
