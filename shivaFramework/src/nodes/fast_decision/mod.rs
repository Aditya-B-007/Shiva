// Fast Decision Engine Sub-Module
//
// WHAT: Re-exports the FastDecisionNode.
// WHY: Clean module hierarchy for the nodes layer.

pub mod node;

pub use node::FastDecisionNode;
