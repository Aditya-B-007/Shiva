// Nodes Core Sub-Module
//
// WHAT: Re-exports the EnvironmentStack shared memory struct.
// WHY: Clean module hierarchy for the nodes layer.

pub mod shared_state;

pub use shared_state::EnvironmentStack;
