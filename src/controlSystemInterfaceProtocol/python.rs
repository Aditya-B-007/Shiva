#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use std::sync::Arc;
#[cfg(feature = "python")]
use crate::shm::{CscpSharedMemory, ACTION_DIM, REWARD_DIM, RULE_MASK_DIM, STATE_DIM};

#[cfg(feature = "python")]
#[pyclass]
pub struct PyCscpSharedMemory {
    pub inner: Arc<CscpSharedMemory>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyCscpSharedMemory {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(CscpSharedMemory::new()),
        }
    }

    pub fn step(
        &mut self,
        obs: Vec<f32>,
        rewards: Vec<f32>,
        mask: Vec<u8>,
    ) -> PyResult<u64> {
        let shm_ptr = Arc::as_ptr(&self.inner) as *mut CscpSharedMemory;
        unsafe {
            let shm_ref = &mut *shm_ptr;
            if obs.len() >= STATE_DIM {
                shm_ref.state_stack.rotate_right(STATE_DIM);
                shm_ref.state_stack[..STATE_DIM].copy_from_slice(&obs[..STATE_DIM]);
            }
            if rewards.len() >= REWARD_DIM {
                shm_ref.reward_stack.rotate_right(REWARD_DIM);
                shm_ref.reward_stack[..REWARD_DIM].copy_from_slice(&rewards[..REWARD_DIM]);
            }
            if mask.len() >= RULE_MASK_DIM {
                shm_ref.rule_mask.rotate_right(RULE_MASK_DIM);
                shm_ref.rule_mask[..RULE_MASK_DIM].copy_from_slice(&mask[..RULE_MASK_DIM]);
            }
        }
        Ok(self.inner.inc_sequence())
    }

    pub fn read_actuation(&self) -> PyResult<(Vec<f32>, f32, u32)> {
        Ok((
            self.inner.actuation_out.to_vec(),
            self.inner.confidence,
            self.inner.latency_us,
        ))
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn cscp_connector(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyCscpSharedMemory>()?;
    Ok(())
}
