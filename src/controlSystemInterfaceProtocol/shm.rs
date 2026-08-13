use std::sync::atomic::{AtomicU64, Ordering};
use crate::error::CscpError;

pub const MAGIC_SIGNATURE: u32 = 0x43534350; // "CSCP"
pub const ABI_VERSION: u32 = 0x00020000;    // 0.2.0

pub const STATE_DIM: usize = 16;
pub const ACTION_DIM: usize = 4;
pub const REWARD_DIM: usize = 2;
pub const RULE_MASK_DIM: usize = 8;
pub const HISTORY_LEN: usize = 4;

/// 64-Byte Cache-Aligned Shared Memory Header
#[repr(C, align(64))]
pub struct CscpSharedMemoryHeader {
    pub magic_signature: u32,
    pub abi_version: u32,
    pub sequence_counter: AtomicU64,
    pub timestamp_us: AtomicU64,
    pub terminated: u8,
    pub truncated: u8,
    pub reserved_pad: [u8; 38],
}

/// Complete Zero-Copy Shared Memory Struct (Aligned to 64 bytes)
#[repr(C, align(64))]
pub struct CscpSharedMemory {
    pub header: CscpSharedMemoryHeader,
    pub state_stack: [f32; STATE_DIM * HISTORY_LEN],
    pub action_stack: [f32; ACTION_DIM * HISTORY_LEN],
    pub reward_stack: [f32; REWARD_DIM * HISTORY_LEN],
    pub rule_mask: [u8; RULE_MASK_DIM * HISTORY_LEN],
    pub actuation_out: [f32; ACTION_DIM],
    pub confidence: f32,
    pub latency_us: u32,
}

impl CscpSharedMemory {
    pub fn new() -> Self {
        Self {
            header: CscpSharedMemoryHeader {
                magic_signature: MAGIC_SIGNATURE,
                abi_version: ABI_VERSION,
                sequence_counter: AtomicU64::new(0),
                timestamp_us: AtomicU64::new(0),
                terminated: 0,
                truncated: 0,
                reserved_pad: [0u8; 38],
            },
            state_stack: [0.0f32; STATE_DIM * HISTORY_LEN],
            action_stack: [0.0f32; ACTION_DIM * HISTORY_LEN],
            reward_stack: [0.0f32; REWARD_DIM * HISTORY_LEN],
            rule_mask: [1u8; RULE_MASK_DIM * HISTORY_LEN],
            actuation_out: [0.0f32; ACTION_DIM],
            confidence: 1.0f32,
            latency_us: 0,
        }
    }

    pub fn validate_header(&self) -> Result<(), CscpError> {
        if self.header.magic_signature != MAGIC_SIGNATURE {
            return Err(CscpError::InvalidMagic(self.header.magic_signature));
        }
        if self.header.abi_version != ABI_VERSION {
            return Err(CscpError::AbiVersionMismatch {
                expected: ABI_VERSION,
                actual: self.header.abi_version,
            });
        }
        Ok(())
    }

    pub fn inc_sequence(&self) -> u64 {
        self.header.sequence_counter.fetch_add(1, Ordering::SeqCst) + 1
    }

    pub fn get_sequence(&self) -> u64 {
        self.header.sequence_counter.load(Ordering::SeqCst)
    }

    pub fn set_timestamp(&self, ts: u64) {
        self.header.timestamp_us.store(ts, Ordering::SeqCst);
    }

    pub fn get_timestamp(&self) -> u64 {
        self.header.timestamp_us.load(Ordering::SeqCst)
    }
}

impl Default for CscpSharedMemory {
    fn default() -> Self {
        Self::new()
    }
}
