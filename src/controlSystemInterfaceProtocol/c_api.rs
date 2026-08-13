use std::ptr;
use crate::error::CscpStatusCode;
use crate::shm::{CscpSharedMemory, ACTION_DIM, REWARD_DIM, RULE_MASK_DIM, STATE_DIM};

#[no_mangle]
pub unsafe extern "C" fn cscp_shm_create() -> *mut CscpSharedMemory {
    let shm = Box::new(CscpSharedMemory::new());
    Box::into_raw(shm)
}

#[no_mangle]
pub unsafe extern "C" fn cscp_shm_destroy(shm: *mut CscpSharedMemory) -> CscpStatusCode {
    if shm.is_null() {
        return CscpStatusCode::NullPointer;
    }
    let _ = Box::from_raw(shm);
    CscpStatusCode::Success
}

#[no_mangle]
pub unsafe extern "C" fn cscp_env_step(
    shm: *mut CscpSharedMemory,
    obs: *const f32,
    rewards: *const f32,
    mask: *const u8,
) -> CscpStatusCode {
    if shm.is_null() || obs.is_null() || rewards.is_null() || mask.is_null() {
        return CscpStatusCode::NullPointer;
    }

    let shm_ref = &mut *shm;
    if let Err(e) = shm_ref.validate_header() {
        return e.into();
    }

    let obs_slice = std::slice::from_raw_parts(obs, STATE_DIM);
    let reward_slice = std::slice::from_raw_parts(rewards, REWARD_DIM);
    let mask_slice = std::slice::from_raw_parts(mask, RULE_MASK_DIM);

    shm_ref.state_stack.rotate_right(STATE_DIM);
    shm_ref.state_stack[..STATE_DIM].copy_from_slice(obs_slice);

    shm_ref.reward_stack.rotate_right(REWARD_DIM);
    shm_ref.reward_stack[..REWARD_DIM].copy_from_slice(reward_slice);

    shm_ref.rule_mask.rotate_right(RULE_MASK_DIM);
    shm_ref.rule_mask[..RULE_MASK_DIM].copy_from_slice(mask_slice);

    shm_ref.inc_sequence();
    CscpStatusCode::Success
}

#[no_mangle]
pub unsafe extern "C" fn cscp_env_read_actuation(
    shm: *const CscpSharedMemory,
    out_action: *mut f32,
    out_confidence: *mut f32,
    out_latency_us: *mut u32,
) -> CscpStatusCode {
    if shm.is_null() || out_action.is_null() {
        return CscpStatusCode::NullPointer;
    }

    let shm_ref = &*shm;
    ptr::copy_nonoverlapping(shm_ref.actuation_out.as_ptr(), out_action, ACTION_DIM);

    if !out_confidence.is_null() {
        *out_confidence = shm_ref.confidence;
    }
    if !out_latency_us.is_null() {
        *out_latency_us = shm_ref.latency_us;
    }

    CscpStatusCode::Success
}
