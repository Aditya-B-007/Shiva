pub mod algo;
pub mod bridge;
pub mod c_api;
pub mod env;
pub mod error;
pub mod python;
pub mod shm;

pub use algo::{UniversalAlgorithmEngine, UniversalPayloadContract};
pub use bridge::CscpShivaBridge;
pub use env::EnvironmentManager;
pub use error::{CscpError, CscpStatusCode};
pub use shm::{
    CscpSharedMemory, CscpSharedMemoryHeader, ABI_VERSION, ACTION_DIM, HISTORY_LEN,
    MAGIC_SIGNATURE, REWARD_DIM, RULE_MASK_DIM, STATE_DIM,
};

