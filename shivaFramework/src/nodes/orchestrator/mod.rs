// Orchestrator Engine Sub-Module
//
// WHAT: Re-exports the MothershipOrchestrator.
// WHY: Clean module hierarchy for the nodes layer.

pub mod mothership;

pub use mothership::MothershipOrchestrator;
