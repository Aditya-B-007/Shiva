// Shiva 2.0 — Framework Lifecycle Model
//
// WHAT THIS FILE DOES:
// Defines the formal lifecycle state machine for the Shiva framework runtime.
// Provides states, transitions, and a trait for lifecycle-aware components.
//
// HOW IT DOES IT:
// Uses an enum-based state machine with explicit transition validation.
// Components implementing `LifecycleAware` receive callbacks at each
// lifecycle phase transition.
//
// WHY WE DO THIS:
// Currently, construction and execution responsibilities are mixed together.
// A formal lifecycle ensures that configuration is complete before execution
// begins, resources are properly initialized, and shutdown is orderly.
// This prevents invalid operations (e.g., executing a cycle before
// initialization) and supports graceful degradation.
//
// LIFECYCLE STATES:
// ┌─────────┐     ┌────────────┐     ┌─────────────┐     ┌─────────┐
// │ Created  │────▶│ Configured │────▶│ Initialized │────▶│ Running │
// └─────────┘     └────────────┘     └─────────────┘     └────┬────┘
//                                                              │
//                                          ┌─────────┐        │
//                                          │ Stopped │◀───────┘
//                                          └────┬────┘
//                                               │
//                                          ┌────▼─────┐
//                                          │ ShutDown │
//                                          └──────────┘

use crate::framework::error::{RuntimeError, ShivaError};

/// Lifecycle states for the Shiva framework runtime.
///
/// WHAT: Enumerates every valid state the framework can occupy.
/// HOW: State transitions are validated by `LifecycleState::can_transition_to()`.
/// WHY: Prevents operations from being attempted in invalid states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LifecycleState {
    /// The runtime has been constructed but not yet configured.
    Created,
    /// Configuration has been applied and validated.
    Configured,
    /// All subsystems have been initialized and are ready to execute.
    Initialized,
    /// The runtime is actively executing control cycles.
    Running,
    /// Execution has been stopped; can be restarted or shut down.
    Stopped,
    /// The runtime has been fully shut down and cannot be reused.
    ShutDown,
}

impl LifecycleState {
    /// Checks whether a transition from `self` to `target` is valid.
    ///
    /// WHAT: Validates lifecycle state transitions.
    /// HOW: Encodes the state machine's adjacency rules.
    /// WHY: Prevents illegal transitions that could leave the runtime in an
    ///      inconsistent state.
    pub fn can_transition_to(&self, target: &LifecycleState) -> bool {
        matches!(
            (self, target),
            (LifecycleState::Created, LifecycleState::Configured)
                | (LifecycleState::Configured, LifecycleState::Initialized)
                | (LifecycleState::Initialized, LifecycleState::Running)
                | (LifecycleState::Running, LifecycleState::Stopped)
                | (LifecycleState::Stopped, LifecycleState::Running)
                | (LifecycleState::Stopped, LifecycleState::ShutDown)
                // Allow emergency shutdown from any active state
                | (LifecycleState::Running, LifecycleState::ShutDown)
                | (LifecycleState::Initialized, LifecycleState::ShutDown)
                | (LifecycleState::Configured, LifecycleState::ShutDown)
        )
    }

    /// Attempts to transition to the target state, returning an error if invalid.
    ///
    /// WHAT: Performs a validated state transition.
    /// HOW: Checks `can_transition_to()` and returns the new state or an error.
    /// WHY: Combines validation and transition into a single atomic operation.
    pub fn transition_to(self, target: LifecycleState) -> Result<LifecycleState, ShivaError> {
        if self.can_transition_to(&target) {
            Ok(target)
        } else {
            Err(ShivaError::Runtime(RuntimeError::InvalidLifecycleTransition {
                from: format!("{:?}", self),
                to: format!("{:?}", target),
            }))
        }
    }
}

impl std::fmt::Display for LifecycleState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LifecycleState::Created => write!(f, "Created"),
            LifecycleState::Configured => write!(f, "Configured"),
            LifecycleState::Initialized => write!(f, "Initialized"),
            LifecycleState::Running => write!(f, "Running"),
            LifecycleState::Stopped => write!(f, "Stopped"),
            LifecycleState::ShutDown => write!(f, "ShutDown"),
        }
    }
}

/// Trait for components that participate in the framework lifecycle.
///
/// WHAT: Callback interface for lifecycle phase transitions.
/// HOW: Implementors receive notifications when the runtime transitions states.
/// WHY: Allows nodes, adapters, and other components to perform phase-specific
///      setup and teardown (e.g., opening connections, allocating buffers,
///      flushing logs).
pub trait LifecycleAware {
    /// Called when the component is being configured.
    ///
    /// WHAT: Receives configuration parameters.
    /// WHY: Allows the component to validate and store its configuration.
    fn on_configure(&mut self, config: &crate::config::ShivaConfig) -> Result<(), ShivaError>;

    /// Called when the component should initialize its internal state.
    ///
    /// WHAT: Performs one-time initialization (e.g., opening connections).
    /// WHY: Separates construction from initialization for deterministic startup.
    fn on_initialize(&mut self) -> Result<(), ShivaError>;

    /// Called when execution is about to begin.
    ///
    /// WHAT: Prepares the component for active execution cycles.
    /// WHY: Allows final pre-execution setup (e.g., starting timers).
    fn on_start(&mut self) -> Result<(), ShivaError>;

    /// Called when execution is being paused or stopped.
    ///
    /// WHAT: Pauses the component's active operations.
    /// WHY: Allows orderly suspension without full teardown.
    fn on_stop(&mut self) -> Result<(), ShivaError>;

    /// Called when the runtime is being fully shut down.
    ///
    /// WHAT: Releases all resources held by the component.
    /// WHY: Ensures clean teardown (e.g., closing connections, flushing buffers).
    fn on_shutdown(&mut self) -> Result<(), ShivaError>;
}
