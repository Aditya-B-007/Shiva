use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use crate::error::CscpError;
use crate::shm::{CscpSharedMemory, ACTION_DIM, REWARD_DIM, RULE_MASK_DIM, STATE_DIM};

pub struct EnvironmentManager {
    shm: Arc<CscpSharedMemory>,
}

impl EnvironmentManager {
    pub fn new(shm: Arc<CscpSharedMemory>) -> Self {
        shm.validate_header().expect("SHM validation failed");
        Self { shm }
    }

    pub fn step(
        &mut self,
        obs: &[f32],
        rewards: &[f32],
        mask: &[u8],
    ) -> Result<u64, CscpError> {
        if obs.len() < STATE_DIM {
            return Err(CscpError::BufferTooSmall {
                provided: obs.len(),
                required: STATE_DIM,
            });
        }
        if rewards.len() < REWARD_DIM {
            return Err(CscpError::BufferTooSmall {
                provided: rewards.len(),
                required: REWARD_DIM,
            });
        }
        if mask.len() < RULE_MASK_DIM {
            return Err(CscpError::BufferTooSmall {
                provided: mask.len(),
                required: RULE_MASK_DIM,
            });
        }

        let shm_ptr = Arc::as_ptr(&self.shm) as *mut CscpSharedMemory;
        unsafe {
            // Shift history stacks
            let shm_ref = &mut *shm_ptr;
            shm_ref.state_stack.rotate_right(STATE_DIM);
            shm_ref.state_stack[..STATE_DIM].copy_from_slice(&obs[..STATE_DIM]);

            shm_ref.reward_stack.rotate_right(REWARD_DIM);
            shm_ref.reward_stack[..REWARD_DIM].copy_from_slice(&rewards[..REWARD_DIM]);

            shm_ref.rule_mask.rotate_right(RULE_MASK_DIM);
            shm_ref.rule_mask[..RULE_MASK_DIM].copy_from_slice(&mask[..RULE_MASK_DIM]);
        }

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_micros() as u64;
        self.shm.set_timestamp(now);

        let seq = self.shm.inc_sequence();
        Ok(seq)
    }

    pub fn read_actuation(&self) -> ([f32; ACTION_DIM], f32, u32) {
        (
            self.shm.actuation_out,
            self.shm.confidence,
            self.shm.latency_us,
        )
    }
}
