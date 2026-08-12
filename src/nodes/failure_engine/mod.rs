// Failure Engine Sub-Module
//
// WHAT: Re-exports the FailureEngineNode.
// WHY: Clean module hierarchy for the nodes layer.

pub mod node;

pub use node::FailureEngineNode;
