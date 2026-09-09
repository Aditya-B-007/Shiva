// GuardRail Engine Sub-Module
//
// WHAT: Re-exports the GuardRailNode.
// WHY: Clean module hierarchy for the nodes layer.

pub mod node;

pub use node::GuardRailNode;
