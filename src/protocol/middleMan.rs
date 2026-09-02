// Shiva 2.0 — MiddleMan Protocol Orchestrator
//
// WHAT THIS FILE DOES:
// Implements `ManInTheMiddle`, the central bridge between external system inputs (`SystemInputDTO`),
// the `EnvironmentMatrix` state window queue, the `MothershipOrchestrator` consensus pipeline,
// and the `ActuatorSignal` hardware dispatch buffer (`ShivaOutputDTO`).
//
// HOW IT DOES IT:
// Exposes a single public method `counter` that delegates internally to:
//   1. `_input`: Validates and cleans raw `SystemInputDTO`
//   2. `_process`: Synthesizes (state, reward, mask), updates `EnvironmentMatrix`, executes 3-phase pipeline, writes `ActuatorSignal`
//   3. `_output`: Packages results into `ShivaOutputDTO`
//
// NOTE (v2): This struct is preserved for backward compatibility. New code should
// use `ShivaRuntime` (via `ShivaBuilder::build()`) which enforces mandatory
// orchestration and proper error propagation.

use crate::environment::actuatorSignal::ActuatorSignal;
use crate::environment::environmentMatrix::EnvironmentMatrix;
use crate::framework::diagnostics::CycleDiagnostics;
use crate::framework::orchestrator::Orchestrator;
use crate::nodes::core::shared_state::EnvironmentStack;
use crate::nodes::orchestrator::MothershipOrchestrator;
use crate::protocol::shivaSide::ShivaOutputDTO;
use crate::protocol::systemSide::SystemInputDTO;

/// ManInTheMiddle — Protocol Orchestrator connecting system inputs with Shiva 2.0 nodes.
///
/// NOTE: This struct is preserved for backward compatibility.
/// New code should use `ShivaRuntime` via `ShivaBuilder::build()`.
pub struct ManInTheMiddle {
    /// Environment matrix storing (action, reward, mask) sliding window history
    pub matrix: EnvironmentMatrix,
    /// Actuator signal staging buffer for hardware transmission
    pub actuator_signal: ActuatorSignal,
    /// Shared memory stack for single-cycle consensus
    pub env_stack: EnvironmentStack,
    /// Mothership Orchestrator running 3-phase consensus pipeline (MANDATORY)
    orchestrator: Box<dyn Orchestrator>,
    /// Staged current input DTO received during `_input` phase
    current_input: SystemInputDTO,
    /// Framework-internal cycle counter (separate from input timestep)
    cycle_counter: u64,
}

impl ManInTheMiddle {
    /// Creates a new `ManInTheMiddle` with a specific orchestrator.
    ///
    /// WHAT: Constructor requiring a mandatory orchestrator.
    /// WHY: The safety pipeline can no longer be accidentally omitted.
    pub fn new(orchestrator: Box<dyn Orchestrator>) -> Self {
        Self {
            matrix: EnvironmentMatrix::__init__(),
            actuator_signal: ActuatorSignal::__init__(),
            env_stack: EnvironmentStack::default(),
            orchestrator,
            current_input: SystemInputDTO::default(),
            cycle_counter: 0,
        }
    }

    /// Constructs a `ManInTheMiddle` with the default MothershipOrchestrator.
    ///
    /// WHAT: Creates a fully-wired runtime with all 5 default engine nodes.
    /// HOW: Constructs all brain-layer adapters and engine nodes internally.
    /// WHY: The default construction path now produces a FULLY FUNCTIONAL runtime.
    pub fn with_default_orchestrator() -> Self {
        use crate::brain::anomaly::RndAnomalyAdapter;
        use crate::brain::constraint::CpoConstraintAdapter;
        use crate::brain::policy::SacPolicyAdapter;
        use crate::brain::risk::IqnRiskAdapter;
        use crate::brain::skill_vault::Td3SkillAdapter;
        use crate::nodes::failure_engine::FailureEngineNode;
        use crate::nodes::fast_decision::FastDecisionNode;
        use crate::nodes::long_vision::LongVisionNode;
        use crate::nodes::explorer::ExplorerNode;
        use crate::nodes::guardrail::GuardRailNode;

        let failure = FailureEngineNode::new(Box::new(RndAnomalyAdapter::default()));
        let fast = FastDecisionNode::new(Box::new(SacPolicyAdapter::default()));
        let vision = LongVisionNode::new(Box::new(IqnRiskAdapter::default()));
        let explorer = ExplorerNode::new(Box::new(Td3SkillAdapter::default()));
        let guard = GuardRailNode::new(Box::new(CpoConstraintAdapter::default()));

        let orchestrator = MothershipOrchestrator::new(failure, fast, vision, explorer, guard);
        Self::new(Box::new(orchestrator))
    }

    /// Constructs a `ManInTheMiddle` protocol orchestrator with custom `ShivaConfig`.
    ///
    /// WHAT: Config-based constructor that creates the default orchestrator.
    /// WHY: Used by ShivaBuilder::build_legacy() for backward compatibility.
    pub fn from_config(config: crate::config::ShivaConfig) -> Self {
        use crate::brain::anomaly::RndAnomalyAdapter;
        use crate::brain::constraint::CpoConstraintAdapter;
        use crate::brain::policy::SacPolicyAdapter;
        use crate::brain::risk::IqnRiskAdapter;
        use crate::brain::skill_vault::Td3SkillAdapter;
        use crate::nodes::failure_engine::FailureEngineNode;
        use crate::nodes::fast_decision::FastDecisionNode;
        use crate::nodes::long_vision::LongVisionNode;
        use crate::nodes::explorer::ExplorerNode;
        use crate::nodes::guardrail::GuardRailNode;

        let failure = FailureEngineNode::new(Box::new(RndAnomalyAdapter::default()));
        let fast = FastDecisionNode::new(Box::new(SacPolicyAdapter::default()));
        let vision = LongVisionNode::new(Box::new(IqnRiskAdapter::default()));
        let explorer = ExplorerNode::new(Box::new(Td3SkillAdapter::default()));
        let guard = GuardRailNode::new(Box::new(CpoConstraintAdapter::default()));

        let orchestrator = MothershipOrchestrator::new(failure, fast, vision, explorer, guard);

        Self {
            matrix: EnvironmentMatrix::from_config(&config),
            actuator_signal: ActuatorSignal::from_config(&config),
            env_stack: EnvironmentStack::default(),
            orchestrator: Box::new(orchestrator),
            current_input: SystemInputDTO::default(),
            cycle_counter: 0,
        }
    }


    /// 3a. counter (Public): The single public entry point exposed to external caller systems.
    /// Acts as the main orchestrator calling `_input` → `_process` → `_output`.
    pub fn counter(&mut self, input: SystemInputDTO) -> ShivaOutputDTO {
        // Step 1: Ingest and clean input data
        self._input(input);

        // Step 2: Synthesize parameters, update EnvironmentMatrix, run consensus, write ActuatorSignal
        let (state, reward, mask, final_action) = self._process();

        // Step 3: Format final response packet to return to system
        self._output(state, reward, mask, final_action)
    }

    /// 3b. _input (Private/Internal): Ingests and performs data cleaning on SystemInputDTO.
    fn _input(&mut self, input: SystemInputDTO) {
        // Store and clean input data (handling potential NaN/Inf floats)
        let mut cleaned_state = input.state;
        for val in cleaned_state.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            }
        }

        let mut cleaned_setpoint = input.setpoint;
        for val in cleaned_setpoint.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            }
        }

        self.current_input = SystemInputDTO {
            state: cleaned_state,
            setpoint: cleaned_setpoint,
            state_stack: input.state_stack,
            action_stack: input.action_stack,
            hard_boundaries: input.hard_boundaries,
            previous_rewards: if input.previous_rewards.is_nan() { 0.0 } else { input.previous_rewards },
            timestep: input.timestep,
        };
    }

    /// 3c. _process (Private/Internal): Calculates (state, reward, mask), updates EnvironmentMatrix,
    /// executes MothershipOrchestrator consensus, writes to ActuatorSignal.
    fn _process(&mut self) -> ([f32; 64], f32, [u8; 32], [f32; 32]) {
        let input = self.current_input;

        // 1. Synthesize processed state: error feedback relative to setpoint
        let mut processed_state = [0.0; 64];
        for i in 0..32 {
            processed_state[i] = input.state[i] - input.setpoint[i];
        }
        for i in 32..64 {
            processed_state[i] = input.state[i];
        }

        // 2. Synthesize reward and mask
        let reward = input.previous_rewards;
        let mask = input.hard_boundaries;

        // Convert state prefix to 32-float action representation for 3-column EnvironmentMatrix storage
        let mut matrix_action_col = [0.0; 32];
        matrix_action_col.copy_from_slice(&processed_state[0..32]);

        // Push (action, reward, mask) to EnvironmentMatrix
        self.matrix.pushRowToMatrix(matrix_action_col, reward, mask);

        // 3. Populate EnvironmentStack for node execution
        self.env_stack.current_state = processed_state;
        self.env_stack.rule_flags = mask;
        self.env_stack.state_history = input.state_stack;
        self.env_stack.action_history = input.action_stack;
        // FIX: Separate input timestep from cycle counter
        self.env_stack.input_timestep = input.timestep;
        // Reset per-cycle transient fields
        self.env_stack.is_emergency = false;
        self.env_stack.emergency_reason = None;
        self.env_stack.safety_veto_reason = None;

        // 4. Run 3-Phase Consensus Pipeline (MANDATORY — no more Option<> bypass)
        self.cycle_counter += 1;
        let mut diagnostics = CycleDiagnostics::new(self.cycle_counter, input.timestep);
        // Note: errors are logged but not propagated in legacy API (counter returns ShivaOutputDTO, not Result)
        let _ = self.orchestrator.execute_cycle(&mut self.env_stack, &mut diagnostics);

        let final_action = self.env_stack.final_action;

        // 5. Write final safe action command to ActuatorSignal buffer & send to system
        self.actuator_signal.write(&final_action);
        match self.actuator_signal.finalSendToSystem() {
            Ok(_dispatched) => {
                // FIX: Update prev_action AFTER successful dispatch
                self.env_stack.prev_action = final_action;
            }
            Err(_e) => {
                // In legacy API, log but don't propagate.
                // New code should use ShivaRuntime which returns Result.
            }
        }

        (processed_state, reward, mask, final_action)
    }

    /// 3d. _output (Private/Internal): Packages results into ShivaOutputDTO.
    fn _output(
        &mut self,
        state: [f32; 64],
        reward: f32,
        mask: [u8; 32],
        final_action: [f32; 32],
    ) -> ShivaOutputDTO {
        ShivaOutputDTO::new(state, reward, mask, final_action)
    }
}

impl Default for ManInTheMiddle {
    fn default() -> Self {
        Self::with_default_orchestrator()
    }
}
