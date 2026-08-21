// Shiva 2.0 — Framework Orchestrator Trait
//
// WHAT THIS FILE DOES:
// Defines the `Orchestrator` trait — the pluggable pipeline coordinator
// interface. The Mothership 5-node architecture becomes one implementation
// of this trait, but users can provide their own orchestration logic.
//
// HOW IT DOES IT:
// Provides a single `execute_cycle()` method that coordinates node execution
// for one complete control cycle, reading from and writing to the
// EnvironmentStack.
//
// WHY WE DO THIS:
// The consensus mechanism should not be hardwired to one particular
// architecture (SAC + IQN + TD3 + RND + CPO). By defining an orchestration
// interface, the framework allows the five-node Mothership to be one
// implementation among many. Users building different control architectures
// can implement their own orchestrators without modifying Shiva core.

use crate::framework::error::ShivaError;
use crate::framework::diagnostics::CycleDiagnostics;
use crate::nodes::core::shared_state::EnvironmentStack;

/// Pluggable pipeline coordinator.
///
/// WHAT: The central coordinator that drives the execution pipeline each cycle.
///
/// HOW: The orchestrator receives the shared `EnvironmentStack`, executes nodes
///      in the correct order, performs consensus merging, and writes the
///      `final_action` to the stack.
///
/// WHY: The Mothership 5-node architecture (SAC, IQN, TD3+z, RND, CPO) is one
///      valid orchestration strategy, not the only one. This trait allows users
///      to implement alternative architectures (e.g., a simpler PID + safety
///      pipeline, or a custom ensemble) while still using the Shiva framework
///      for lifecycle, error handling, hardware dispatch, and safety.
///
/// # Contract
///
/// After `execute_cycle()` returns `Ok(())`:
/// - `env.final_action` MUST contain the action to be dispatched
/// - `env.is_emergency` MUST be set if an emergency was triggered
/// - `env.cycle_counter` SHOULD be incremented
///
/// # Example
///
/// ```rust
/// use shiva::framework::{Orchestrator, ShivaError, CycleDiagnostics};
/// use shiva::nodes::EnvironmentStack;
///
/// struct SimpleOrchestrator;
///
/// impl Orchestrator for SimpleOrchestrator {
///     fn execute_cycle(
///         &self,
///         env: &mut EnvironmentStack,
///         diagnostics: &mut CycleDiagnostics,
///     ) -> Result<(), ShivaError> {
///         // Simple pass-through: final_action = candidate_action
///         env.final_action = env.candidate_action;
///         env.cycle_counter += 1;
///         Ok(())
///     }
/// }
/// ```
pub trait Orchestrator: Send + Sync {
    /// Executes one complete control cycle.
    ///
    /// WHAT: Coordinates the full pipeline execution for a single timestep.
    ///
    /// HOW: Runs all registered nodes in phase order, performs consensus
    ///      merging (Phase 2), applies safety projection (Phase 3), and
    ///      writes the final safe action to `env.final_action`.
    ///
    /// WHY: This is the single method that drives the Shiva execution pipeline.
    ///      By implementing this trait, users can replace the entire orchestration
    ///      strategy while retaining the framework's lifecycle, error handling,
    ///      and hardware dispatch infrastructure.
    ///
    /// # Arguments
    /// * `env` — Mutable reference to the shared EnvironmentStack
    /// * `diagnostics` — Mutable reference to the cycle diagnostics collector
    ///
    /// # Returns
    /// * `Ok(())` — Cycle completed successfully; `env.final_action` is set
    /// * `Err(ShivaError)` — Cycle failed; the runtime should handle the error
    fn execute_cycle(
        &self,
        env: &mut EnvironmentStack,
        diagnostics: &mut CycleDiagnostics,
    ) -> Result<(), ShivaError>;
}
