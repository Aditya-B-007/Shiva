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

use crate::environment::actuatorSignal::ActuatorSignal;
use crate::environment::environmentMatrix::EnvironmentMatrix;
use crate::nodes::core::shared_state::EnvironmentStack;
use crate::nodes::orchestrator::MothershipOrchestrator;
use crate::protocol::shivaSide::ShivaOutputDTO;
use crate::protocol::systemSide::SystemInputDTO;

/// ManInTheMiddle — Protocol Orchestrator connecting system inputs with Shiva 2.0 nodes.
pub struct ManInTheMiddle {
    /// Environment matrix storing (action, reward, mask) sliding window history
    pub matrix: EnvironmentMatrix,
    /// Actuator signal staging buffer for hardware transmission
    pub actuator_signal: ActuatorSignal,
    /// Shared memory stack for single-cycle consensus
    pub env_stack: EnvironmentStack,
    /// Mothership Orchestrator running 3-phase consensus pipeline (optional/pluggable)
    pub orchestrator: Option<MothershipOrchestrator>,
    /// Staged current input DTO received during `_input` phase
    current_input: SystemInputDTO,
}

impl ManInTheMiddle {
    /// Creates a new `ManInTheMiddle` protocol orchestrator.
    pub fn new(orchestrator: Option<MothershipOrchestrator>) -> Self {
        Self {
            matrix: EnvironmentMatrix::__init__(),
            actuator_signal: ActuatorSignal::__init__(),
            env_stack: EnvironmentStack::default(),
            orchestrator,
            current_input: SystemInputDTO::default(),
        }
    }

    /// Alias constructor matching requested naming convention.
    pub fn __init__() -> Self {
        Self::new(None)
    }

    /// Constructs a `ManInTheMiddle` protocol orchestrator with custom `ShivaConfig`.
    pub fn from_config(config: crate::config::ShivaConfig) -> Self {
        Self {
            matrix: EnvironmentMatrix::from_config(&config),
            actuator_signal: ActuatorSignal::from_config(&config),
            env_stack: EnvironmentStack::default(),
            orchestrator: None,
            current_input: SystemInputDTO::default(),
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
        self.env_stack.cycle_counter = input.timestep;

        // 4. Run 3-Phase Consensus Pipeline if MothershipOrchestrator is attached
        if let Some(ref orchestrator) = self.orchestrator {
            orchestrator.execute_cycle(&mut self.env_stack);
        } else {
            // Fallback baseline: pass-through candidate action or state delta
            self.env_stack.final_action = matrix_action_col;
        }

        let final_action = self.env_stack.final_action;

        // 5. Write final safe action command to ActuatorSignal buffer & send to system
        self.actuator_signal.write(&final_action);
        let _ = self.actuator_signal.finalSendToSystem();

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
        Self::__init__()
    }
}
