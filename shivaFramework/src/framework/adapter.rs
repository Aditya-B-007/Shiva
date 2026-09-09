// Shiva 2.0 — Framework Input/Output Adapter Traits
//
// WHAT THIS FILE DOES:
// Defines `InputAdapter` and `OutputAdapter` traits that decouple the
// framework core from specific hardware transports (CAN, EtherCAT, ROS,
// simulation, serial, custom interfaces).
//
// HOW IT DOES IT:
// `InputAdapter` converts raw `SystemInputDTO` data into cleaned internal
// state. `OutputAdapter` dispatches the final safe action to a hardware
// transport. Both are trait objects, allowing runtime polymorphism.
//
// WHY WE DO THIS:
// `ManInTheMiddle` currently writes directly to `ActuatorSignal` and calls
// `finalSendToSystem()`, coupling the core runtime to one specific hardware
// dispatch mechanism. By abstracting input/output through traits, the
// framework becomes reusable across different hardware transports.

use crate::framework::error::ShivaError;
use crate::framework::dimensions::{StateVector, ActionVector, RuleFlagVector};
use crate::protocol::systemSide::SystemInputDTO;

/// Result of an input adaptation operation.
///
/// WHAT: The cleaned and transformed input data ready for pipeline processing.
/// WHY: Separates raw input cleaning (NaN/Inf sanitization) from pipeline logic.
#[derive(Debug, Clone, Copy)]
pub struct AdaptedInput {
    /// Cleaned and processed state observation vector.
    pub processed_state: StateVector,
    /// Cleaned setpoint / target state.
    pub setpoint: ActionVector,
    /// State history window.
    pub state_history: StateVector,
    /// Action history window.
    pub action_history: ActionVector,
    /// Hardware safety interlock bitmask.
    pub rule_flags: RuleFlagVector,
    /// Reward feedback scalar from previous cycle.
    pub reward: f32,
    /// External environment timestep.
    pub timestep: u64,
}

/// Result of an output dispatch operation.
///
/// WHAT: Confirmation that the action was successfully dispatched to hardware.
/// WHY: Allows the runtime to confirm dispatch and update internal state
///      (e.g., prev_action) only after successful transmission.
#[derive(Debug, Clone)]
pub struct DispatchResult {
    /// The action values that were actually dispatched.
    pub dispatched_action: ActionVector,
}

/// Converts external system data into the framework's internal state representation.
///
/// WHAT: Trait for input preprocessing and cleaning.
///
/// HOW: Receives raw `SystemInputDTO` from the external system and produces
///      a cleaned `AdaptedInput` with NaN/Inf sanitization, state error
///      computation (state - setpoint), and any other preprocessing.
///
/// WHY: Different hardware sources may require different input processing
///      (e.g., ROS message parsing, CAN frame decoding, simulation bridge).
///      Abstracting this behind a trait allows the framework to support
///      multiple input sources without modifying core logic.
pub trait InputAdapter: Send + Sync {
    /// Adapts raw system input into cleaned framework input.
    ///
    /// # Arguments
    /// * `raw` — The raw input DTO from the external system
    ///
    /// # Returns
    /// * `Ok(AdaptedInput)` — Successfully cleaned and transformed input
    /// * `Err(ShivaError)` — Input adaptation failed
    fn adapt_input(&self, raw: &SystemInputDTO) -> Result<AdaptedInput, ShivaError>;
}

/// Dispatches the final safe action to a hardware transport.
///
/// WHAT: Trait for hardware output dispatch.
///
/// HOW: Receives the final safe action vector and transmits it to the
///      physical actuator interface. The implementation handles transport-
///      specific details (CAN bus transmission, serial write, ROS publish,
///      simulation API call, etc.).
///
/// WHY: `ManInTheMiddle` currently calls `ActuatorSignal::finalSendToSystem()`
///      directly, coupling the core to one specific dispatch mechanism.
///      This trait allows users to implement custom hardware transports
///      without modifying the framework core.
///
/// # Contract
///
/// Implementations MUST:
/// 1. Clamp values to configured actuator limits before transmission
/// 2. Return `Err` if dispatch fails — never silently discard errors
/// 3. Be safe to call from the real-time execution path
pub trait OutputAdapter: Send + Sync {
    /// Dispatches the final safe action to hardware.
    ///
    /// # Arguments
    /// * `action` — The final safe action vector to dispatch
    ///
    /// # Returns
    /// * `Ok(DispatchResult)` — Action was successfully dispatched
    /// * `Err(ShivaError)` — Dispatch failed
    fn dispatch(&mut self, action: &ActionVector) -> Result<DispatchResult, ShivaError>;
}
