// Shiva 2.0 — Environment Sub-Module
//
// WHAT THIS MODULE DOES:
// Manages the EnvironmentMatrix state window queue and ActuatorSignal hardware dispatch buffer.
//
// HOW IT DOES IT:
// - `environmentMatrix`: Fixed-capacity sliding window matrix with node access control policy.
// - `actuatorSignal`: Staging and dispatch interface for hardware motor signals.

pub mod environmentMatrix;
pub mod actuatorSignal;

pub use environmentMatrix::{EnvironmentMatrix, MatrixRow, NodeType, MatrixAccessError};
pub use actuatorSignal::ActuatorSignal;
