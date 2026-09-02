// Shiva 2.0 — Default Input Adapter
//
// WHAT THIS FILE DOES:
// Provides the default InputAdapter implementation that cleans and transforms
// raw SystemInputDTO data into the framework's internal state representation.
//
// HOW IT DOES IT:
// Sanitizes NaN/Inf values, computes goal-conditioned state error
// (state - setpoint), and packages the result into an AdaptedInput.
//
// WHY WE DO THIS:
// Extracts the input cleaning logic from ManInTheMiddle::_input() into a
// pluggable adapter, allowing users to implement custom input processing
// for different hardware sources (ROS, CAN, simulation, etc.).

use crate::framework::adapter::{AdaptedInput, InputAdapter};
use crate::framework::error::ShivaError;
use crate::protocol::systemSide::SystemInputDTO;

/// Default InputAdapter that sanitizes NaN/Inf and computes state error.
///
/// WHAT: Cleans raw input data and computes goal-conditioned processed state.
/// HOW: Replaces NaN/Inf with 0.0, then computes state[i] - setpoint[i] for
///      the first 32 elements and passes through the remaining 32.
/// WHY: Ensures the pipeline never receives invalid floating-point values.
pub struct DefaultInputAdapter;

impl DefaultInputAdapter {
    pub fn new() -> Self {
        Self
    }
}

impl Default for DefaultInputAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl InputAdapter for DefaultInputAdapter {
    fn adapt_input(&self, raw: &SystemInputDTO) -> Result<AdaptedInput, ShivaError> {
        // Clean state: replace NaN/Inf with 0.0
        let mut cleaned_state = raw.state;
        for val in cleaned_state.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            }
        }

        // Clean setpoint: replace NaN/Inf with 0.0
        let mut cleaned_setpoint = raw.setpoint;
        for val in cleaned_setpoint.iter_mut() {
            if val.is_nan() || val.is_infinite() {
                *val = 0.0;
            }
        }

        // Compute goal-conditioned processed state:
        // First 32 elements: error = state - setpoint
        // Last 32 elements: pass-through
        let mut processed_state = [0.0f32; 64];
        for i in 0..32 {
            processed_state[i] = cleaned_state[i] - cleaned_setpoint[i];
        }
        for i in 32..64 {
            processed_state[i] = cleaned_state[i];
        }

        // Clean reward
        let reward = if raw.previous_rewards.is_nan() {
            0.0
        } else {
            raw.previous_rewards
        };

        Ok(AdaptedInput {
            processed_state,
            setpoint: cleaned_setpoint,
            state_history: raw.state_stack,
            action_history: raw.action_stack,
            rule_flags: raw.hard_boundaries,
            reward,
            timestep: raw.timestep,
        })
    }
}
