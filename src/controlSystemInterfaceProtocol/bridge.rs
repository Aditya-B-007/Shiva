// Shiva 2.0 — Control System Context Protocol (CSCP) Shiva Bridge
//
// WHAT THIS FILE DOES:
// Bridges external CSCP shared memory payloads (CscpSharedMemory / UniversalPayloadContract)
// with Shiva 2.0's internal 64-byte aligned EnvironmentStack and 5-Node Mothership Ensemble.
//
// HOW IT DOES IT:
// Provides zero-allocation synchronization methods to extract CSCP observation, action, and rule stack data
// into `EnvironmentStack`, and to commit Shiva Orchestrator consensus decisions back to CSCP actuation arrays.
//
// WHY WE DO THIS:
// Enables Shiva's 5-Node Mothership Ensemble to act as Side B (universal RL / control engine) for any
// physical robot or simulator speaking the C-ABI Control-System-Context-Protocol without high-latency copies.

use std::sync::Arc;
use crate::controlSystemInterfaceProtocol::error::CscpError;
use crate::controlSystemInterfaceProtocol::shm::{
    CscpSharedMemory, ACTION_DIM, HISTORY_LEN, REWARD_DIM, RULE_MASK_DIM, STATE_DIM,
};
use crate::nodes::core::EnvironmentStack;

/// Zero-copy adapter bridging CSCP shared memory and Shiva internal EnvironmentStack
pub struct CscpShivaBridge {
    shm: Arc<CscpSharedMemory>,
}

impl CscpShivaBridge {
    pub fn new(shm: Arc<CscpSharedMemory>) -> Result<Self, CscpError> {
        shm.validate_header()?;
        Ok(Self { shm })
    }

    /// Access underlying CSCP shared memory reference
    pub fn shm(&self) -> &Arc<CscpSharedMemory> {
        &self.shm
    }

    /// Syncs CSCP payload into Shiva's internal 64-byte aligned EnvironmentStack.
    ///
    /// Copies state vectors, action history, and safety rule bitmasks into Shiva's
    /// `EnvironmentStack` ready for execution through `MothershipOrchestrator`.
    pub fn sync_to_environment_stack(&self, env_stack: &mut EnvironmentStack) {
        // Copy latest state (first STATE_DIM elements) into current_state
        env_stack.current_state[..STATE_DIM].copy_from_slice(&self.shm.state_stack[..STATE_DIM]);

        // Copy latest executed action into prev_action
        env_stack.prev_action[..ACTION_DIM].copy_from_slice(&self.shm.action_stack[..ACTION_DIM]);

        // Copy rule flags
        env_stack.rule_flags[..RULE_MASK_DIM].copy_from_slice(&self.shm.rule_mask[..RULE_MASK_DIM]);

        // Copy flattened state history (up to STATE_DIM * HISTORY_LEN = 64 floats)
        let state_len = (STATE_DIM * HISTORY_LEN).min(env_stack.state_history.len());
        env_stack.state_history[..state_len].copy_from_slice(&self.shm.state_stack[..state_len]);

        // Copy flattened action history (up to ACTION_DIM * HISTORY_LEN = 16 floats into 32 float history)
        let action_len = (ACTION_DIM * HISTORY_LEN).min(env_stack.action_history.len());
        env_stack.action_history[..action_len].copy_from_slice(&self.shm.action_stack[..action_len]);
    }

    /// Commits Shiva Orchestrator's final consensus action back into CSCP shared memory.
    pub fn commit_final_action(
        &mut self,
        final_action: &[f32; 32],
        confidence: f32,
        latency_us: u32,
    ) -> Result<(), CscpError> {
        let shm_ptr = Arc::as_ptr(&self.shm) as *mut CscpSharedMemory;
        unsafe {
            let shm_ref = &mut *shm_ptr;
            shm_ref.actuation_out.copy_from_slice(&final_action[..ACTION_DIM]);
            shm_ref.confidence = confidence;
            shm_ref.latency_us = latency_us;

            // Push actuation into action_stack history
            shm_ref.action_stack.rotate_right(ACTION_DIM);
            shm_ref.action_stack[..ACTION_DIM].copy_from_slice(&final_action[..ACTION_DIM]);
        }
        Ok(())
    }
}
