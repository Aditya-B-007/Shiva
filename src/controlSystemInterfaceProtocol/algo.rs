use std::sync::Arc;
use crate::error::CscpError;
use crate::shm::{CscpSharedMemory, ACTION_DIM, HISTORY_LEN, REWARD_DIM, RULE_MASK_DIM, STATE_DIM};

#[derive(Debug, Clone)]
pub struct UniversalPayloadContract {
    pub sequence: u64,
    pub timestamp_us: u64,
    pub terminated: bool,
    pub truncated: bool,
    pub state_stack: [f32; STATE_DIM * HISTORY_LEN],
    pub action_stack: [f32; ACTION_DIM * HISTORY_LEN],
    pub reward_stack: [f32; REWARD_DIM * HISTORY_LEN],
    pub rule_mask: [u8; RULE_MASK_DIM * HISTORY_LEN],
}

pub struct UniversalAlgorithmEngine {
    shm: Arc<CscpSharedMemory>,
}

impl UniversalAlgorithmEngine {
    pub fn new(shm: Arc<CscpSharedMemory>) -> Self {
        shm.validate_header().expect("SHM validation failed");
        Self { shm }
    }

    pub fn read_payload(&self) -> UniversalPayloadContract {
        UniversalPayloadContract {
            sequence: self.shm.get_sequence(),
            timestamp_us: self.shm.get_timestamp(),
            terminated: self.shm.header.terminated != 0,
            truncated: self.shm.header.truncated != 0,
            state_stack: self.shm.state_stack,
            action_stack: self.shm.action_stack,
            reward_stack: self.shm.reward_stack,
            rule_mask: self.shm.rule_mask,
        }
    }

    pub fn write_actuation(
        &mut self,
        actuation: &[f32],
        confidence: f32,
        latency_us: u32,
    ) -> Result<(), CscpError> {
        if actuation.len() < ACTION_DIM {
            return Err(CscpError::BufferTooSmall {
                provided: actuation.len(),
                required: ACTION_DIM,
            });
        }

        let shm_ptr = Arc::as_ptr(&self.shm) as *mut CscpSharedMemory;
        unsafe {
            let shm_ref = &mut *shm_ptr;
            shm_ref.actuation_out.copy_from_slice(&actuation[..ACTION_DIM]);
            shm_ref.confidence = confidence;
            shm_ref.latency_us = latency_us;

            // Push actuation into action_stack history
            shm_ref.action_stack.rotate_right(ACTION_DIM);
            shm_ref.action_stack[..ACTION_DIM].copy_from_slice(&actuation[..ACTION_DIM]);
        }

        Ok(())
    }
}
