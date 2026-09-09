// Shiva 2.0 — Framework Execution Context
//
// WHAT THIS FILE DOES:
// Defines `ExecutionContext`, the per-cycle data envelope passed through
// the execution pipeline. Wraps the `EnvironmentStack` and provides
// additional context (cycle ID, input timestep, diagnostics) that nodes
// can use during execution.
//
// HOW IT DOES IT:
// Holds a mutable reference to the `EnvironmentStack` along with cycle
// metadata and a diagnostics collector. Nodes receive this context
// instead of raw `EnvironmentStack` references.
//
// WHY WE DO THIS:
// Separates framework-level concerns (cycle counting, timestep tracking,
// diagnostics) from the shared state buffer (EnvironmentStack). This
// ensures that cycle_counter is managed by the framework (not conflated
// with input timestep), and diagnostics are collected uniformly.

use crate::framework::diagnostics::CycleDiagnostics;
use crate::nodes::core::shared_state::EnvironmentStack;

/// Per-cycle execution context passed through the Shiva pipeline.
///
/// WHAT: Envelope containing the shared state, cycle metadata, and diagnostics.
///
/// HOW: Created by the runtime at the start of each cycle. Passed to the
///      orchestrator and individual nodes. Consumed at the end of the cycle.
///
/// WHY: Provides nodes with everything they need for execution without
///      exposing internal framework state. Separates the external timestep
///      from the internal cycle counter, and collects diagnostics uniformly.
pub struct ExecutionContext<'a> {
    /// Mutable reference to the shared EnvironmentStack.
    ///
    /// WHAT: The C-contiguous shared memory buffer for all node I/O.
    /// WHY: Nodes read inputs from and write outputs to this structure.
    pub env: &'a mut EnvironmentStack,

    /// Framework-internal cycle counter (monotonically increasing).
    ///
    /// WHAT: The number of complete execution cycles the framework has run.
    /// WHY: Separate from `input_timestep` — the framework counts its own
    ///      execution cycles independently of the external environment clock.
    pub cycle_id: u64,

    /// External environment timestep from the input DTO.
    ///
    /// WHAT: The timestamp/step-index provided by the external system.
    /// WHY: Preserves the external clock for correlation with external telemetry.
    ///      NOT used for framework-internal cycle counting.
    pub input_timestep: u64,

    /// Diagnostics collector for this cycle.
    ///
    /// WHAT: Accumulates per-node timings, emergency/veto reasons, and
    ///       consensus weights during pipeline execution.
    /// WHY: Provides observability data to framework users after each cycle.
    pub diagnostics: CycleDiagnostics,
}

impl<'a> ExecutionContext<'a> {
    /// Creates a new execution context for a cycle.
    ///
    /// WHAT: Initializes the context with the given stack, cycle ID, and timestep.
    /// WHY: Called by the runtime at the start of each `counter()` invocation.
    pub fn new(env: &'a mut EnvironmentStack, cycle_id: u64, input_timestep: u64) -> Self {
        Self {
            env,
            cycle_id,
            input_timestep,
            diagnostics: CycleDiagnostics::new(cycle_id, input_timestep),
        }
    }
}
