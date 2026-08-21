// Shiva 2.0 — ShivaRuntime
//
// WHAT THIS FILE DOES:
// Defines `ShivaRuntime`, the primary framework entry point that replaces
// `ManInTheMiddle` as the recommended public API. It manages the complete
// lifecycle: input adaptation, orchestration, output dispatch, and
// state lifecycle correctness.
//
// HOW IT DOES IT:
// Holds a configured orchestrator (trait object), input/output adapters,
// shared EnvironmentStack, EnvironmentMatrix, lifecycle state, and
// diagnostics. The single public method `step()` processes one cycle.
//
// WHY WE DO THIS:
// `ManInTheMiddle` allowed the orchestrator to be `None`, bypassing the
// safety pipeline. `ShivaRuntime` enforces mandatory orchestration and
// proper lifecycle management by construction.
//
// KEY DIFFERENCES FROM ManInTheMiddle:
// 1. Orchestrator is MANDATORY — cannot be None
// 2. Hardware dispatch goes through OutputAdapter trait — not hardwired
// 3. Input cleaning goes through InputAdapter trait — pluggable
// 4. prev_action is updated AFTER successful dispatch
// 5. cycle_counter is separate from input.timestep
// 6. Errors are propagated, not discarded
// 7. CycleDiagnostics are collected and returned

use crate::config::ShivaConfig;
use crate::environment::environmentMatrix::EnvironmentMatrix;
use crate::framework::adapter::{InputAdapter, OutputAdapter};
use crate::framework::diagnostics::CycleDiagnostics;
use crate::framework::error::{RuntimeError, ShivaError};
use crate::framework::lifecycle::LifecycleState;
use crate::framework::orchestrator::Orchestrator;
use crate::nodes::core::shared_state::EnvironmentStack;
use crate::protocol::shivaSide::ShivaOutputDTO;
use crate::protocol::systemSide::SystemInputDTO;

/// ShivaRuntime — The primary framework entry point.
///
/// WHAT: Manages the complete control cycle from input to dispatch.
///
/// HOW: Orchestrates input adaptation → pipeline execution → output dispatch
///      with mandatory safety/orchestration and lifecycle enforcement.
///
/// WHY: Replaces `ManInTheMiddle` as the recommended public API. Guarantees
///      that the safety pipeline is always active, errors are propagated,
///      and the state lifecycle contract is enforced.
pub struct ShivaRuntime {
    /// Framework configuration.
    config: ShivaConfig,
    /// The pipeline coordinator (mandatory — cannot be None).
    orchestrator: Box<dyn Orchestrator>,
    /// Hardware output dispatch adapter.
    output_adapter: Box<dyn OutputAdapter>,
    /// Input preprocessing adapter.
    input_adapter: Box<dyn InputAdapter>,
    /// Shared memory stack for single-cycle consensus.
    env_stack: EnvironmentStack,
    /// Sliding window state store.
    matrix: EnvironmentMatrix,
    /// Current lifecycle state.
    lifecycle: LifecycleState,
    /// Framework-internal cycle counter (monotonically increasing).
    cycle_counter: u64,
    /// Diagnostics from the most recent cycle.
    last_diagnostics: CycleDiagnostics,
}

impl ShivaRuntime {
    /// Creates a new ShivaRuntime with the given configuration and components.
    ///
    /// WHAT: Constructor for the fully-wired runtime.
    /// HOW: Accepts pre-constructed orchestrator and adapters.
    /// WHY: Construction is separate from execution — follows lifecycle model.
    pub fn new(
        config: ShivaConfig,
        orchestrator: Box<dyn Orchestrator>,
        output_adapter: Box<dyn OutputAdapter>,
        input_adapter: Box<dyn InputAdapter>,
    ) -> Self {
        Self {
            matrix: EnvironmentMatrix::from_config(&config),
            env_stack: EnvironmentStack::default(),
            lifecycle: LifecycleState::Initialized,
            cycle_counter: 0,
            last_diagnostics: CycleDiagnostics::default(),
            config,
            orchestrator,
            output_adapter,
            input_adapter,
        }
    }

    /// Executes a single control cycle: input → pipeline → output.
    ///
    /// WHAT: The main entry point for processing one timestep.
    ///
    /// HOW:
    /// 1. Adapts raw input (NaN cleaning, state error computation)
    /// 2. Updates EnvironmentMatrix with the new state
    /// 3. Populates EnvironmentStack for node execution
    /// 4. Runs the orchestrator's 3-phase pipeline
    /// 5. Dispatches the final safe action via OutputAdapter
    /// 6. Updates prev_action AFTER successful dispatch
    /// 7. Collects and stores diagnostics
    ///
    /// WHY: Single method covering the complete control loop with proper
    ///      error propagation and state lifecycle enforcement.
    pub fn step(&mut self, input: SystemInputDTO) -> Result<ShivaOutputDTO, ShivaError> {
        // 1. Adapt raw input
        let adapted = self.input_adapter.adapt_input(&input)?;

        // 2. Push to EnvironmentMatrix
        let mut matrix_action_col = [0.0f32; 32];
        matrix_action_col.copy_from_slice(&adapted.processed_state[0..32]);
        self.matrix.pushRowToMatrix(matrix_action_col, adapted.reward, adapted.rule_flags);

        // 3. Populate EnvironmentStack
        self.env_stack.current_state = adapted.processed_state;
        self.env_stack.rule_flags = adapted.rule_flags;
        self.env_stack.state_history = adapted.state_history;
        self.env_stack.action_history = adapted.action_history;
        // Preserve external timestep separately from cycle_counter
        self.env_stack.input_timestep = adapted.timestep;
        // Reset per-cycle transient fields
        self.env_stack.is_emergency = false;
        self.env_stack.emergency_reason = None;
        self.env_stack.safety_veto_reason = None;

        // 4. Run orchestrator pipeline
        self.cycle_counter += 1;
        let mut diagnostics = CycleDiagnostics::new(self.cycle_counter, adapted.timestep);
        self.orchestrator.execute_cycle(&mut self.env_stack, &mut diagnostics)?;

        let final_action = self.env_stack.final_action;

        // 5. Dispatch via OutputAdapter (propagate errors!)
        let dispatch_result = self.output_adapter.dispatch(&final_action)?;

        // 6. Update prev_action AFTER successful dispatch — state lifecycle contract
        self.env_stack.prev_action = dispatch_result.dispatched_action;

        // 7. Store diagnostics
        self.last_diagnostics = diagnostics;

        // 8. Build output DTO
        Ok(ShivaOutputDTO::new(
            adapted.processed_state,
            adapted.reward,
            adapted.rule_flags,
            final_action,
        ))
    }

    /// Returns the diagnostics from the most recent execution cycle.
    pub fn last_diagnostics(&self) -> &CycleDiagnostics {
        &self.last_diagnostics
    }

    /// Returns the current lifecycle state.
    pub fn lifecycle_state(&self) -> LifecycleState {
        self.lifecycle
    }

    /// Returns the framework-internal cycle counter.
    pub fn cycle_counter(&self) -> u64 {
        self.cycle_counter
    }

    /// Returns a reference to the current EnvironmentStack (for inspection/testing).
    pub fn env_stack(&self) -> &EnvironmentStack {
        &self.env_stack
    }

    /// Returns a reference to the framework configuration.
    pub fn config(&self) -> &ShivaConfig {
        &self.config
    }

    /// Backward-compatible entry point matching `ManInTheMiddle::counter()` signature.
    ///
    /// WHY: Allows gradual migration from ManInTheMiddle to ShivaRuntime.
    /// NOTE: Panics on error — prefer `step()` for proper error handling.
    pub fn counter(&mut self, input: SystemInputDTO) -> ShivaOutputDTO {
        self.step(input).expect("ShivaRuntime::counter() failed — use step() for error handling")
    }
}
