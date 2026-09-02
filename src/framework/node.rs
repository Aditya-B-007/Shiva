// Shiva 2.0 — Framework Node Trait
//
// WHAT THIS FILE DOES:
// Defines the `Node` trait — the universal extension point for the Shiva
// framework pipeline. Any component that participates in the execution
// pipeline must implement this trait.
//
// HOW IT DOES IT:
// Provides a trait with `name()`, `phase()`, and `execute()` methods.
// The `Phase` enum determines execution ordering, and `NodeOutcome`
// controls pipeline flow (continue vs. short-circuit).
//
// WHY WE DO THIS:
// External developers need a stable interface for implementing their own
// nodes. The current architecture is tightly coupled to the five built-in
// engines. This trait allows users to add/replace nodes without modifying
// Shiva core, making the architecture truly extensible.

use crate::framework::error::ShivaError;
use crate::nodes::core::shared_state::EnvironmentStack;

/// Pipeline execution phase determining node ordering.
///
/// WHAT: Classifies when a node executes within the 3-phase pipeline.
/// HOW: The orchestrator sorts/groups nodes by phase and executes them in order.
/// WHY: Maintains the deterministic Phase 1 → Phase 2 → Phase 3 execution
///      guarantee while allowing custom phases for user extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    /// Phase 1: Anomaly gate — executes before any policy computation.
    /// Nodes in this phase can short-circuit the entire pipeline.
    AnomalyGate = 0,

    /// Phase 2: Candidate consensus — multiple engines propose actions.
    /// All Phase 2 nodes execute; their outputs are merged by the orchestrator.
    Consensus = 1,

    /// Phase 3: Safety shield — immutable post-pass safety filter.
    /// Nodes in this phase enforce physical constraints and can veto actions.
    SafetyShield = 2,

    /// User-defined phase for custom pipeline extensions.
    /// Custom phases execute after all standard phases (ordered by value).
    Custom(u32),
}

/// Outcome of a single node execution, controlling pipeline flow.
///
/// WHAT: Signals to the orchestrator whether to continue or short-circuit.
/// HOW: Returned by `Node::execute()` after each node processes a cycle.
/// WHY: Enables Phase 1 anomaly detection and Phase 3 veto to halt the
///      pipeline early without the orchestrator needing node-specific knowledge.
#[derive(Debug, Clone)]
pub enum NodeOutcome {
    /// The pipeline should continue to the next node/phase.
    Continue,

    /// The pipeline should short-circuit (skip remaining nodes).
    /// The orchestrator uses the current `EnvironmentStack` state as-is.
    ShortCircuit {
        /// Human-readable reason for the short-circuit (for diagnostics).
        reason: &'static str,
    },
}

/// The fundamental framework extension point.
///
/// WHAT: Trait that all pipeline participants must implement.
///
/// HOW: Nodes are registered with the orchestrator (or builder) and executed
///      in phase order during each control cycle. Each node reads from and
///      writes to the shared `EnvironmentStack`.
///
/// WHY: This is the primary mechanism for extending Shiva. Users implement
///      this trait to add custom anomaly detectors, policy engines, safety
///      filters, or any other pipeline component — without modifying core
///      framework code.
///
/// # Example
///
/// ```rust
/// use shiva::framework::{Node, Phase, NodeOutcome, ShivaError};
/// use shiva::nodes::EnvironmentStack;
///
/// struct MyCustomNode;
///
/// impl Node for MyCustomNode {
///     fn name(&self) -> &str { "MyCustomNode" }
///     fn phase(&self) -> Phase { Phase::Consensus }
///     fn execute(&self, env: &mut EnvironmentStack) -> Result<NodeOutcome, ShivaError> {
///         // Custom logic here
///         Ok(NodeOutcome::Continue)
///     }
/// }
/// ```
pub trait Node: Send + Sync {
    /// Returns the human-readable name of this node.
    ///
    /// WHAT: Identifier used in diagnostics, logging, and error messages.
    /// WHY: Allows the runtime to attribute errors and timings to specific nodes.
    fn name(&self) -> &str;

    /// Returns the pipeline phase this node belongs to.
    ///
    /// WHAT: Determines when this node executes relative to others.
    /// WHY: The orchestrator uses this to maintain the Phase 1 → 2 → 3 ordering.
    fn phase(&self) -> Phase;

    /// Executes this node for one control cycle.
    ///
    /// WHAT: The node's core logic — reads inputs from and writes outputs to
    ///       the shared `EnvironmentStack`.
    ///
    /// HOW: Called by the orchestrator during the appropriate phase. The node
    ///      should read its required inputs from `env`, perform computation,
    ///      and write its outputs back to `env`.
    ///
    /// WHY: This is the single method that makes a component participate in
    ///      the Shiva execution pipeline.
    ///
    /// # Returns
    /// - `Ok(NodeOutcome::Continue)` — pipeline continues to next node
    /// - `Ok(NodeOutcome::ShortCircuit { reason })` — pipeline halts early
    /// - `Err(ShivaError)` — node encountered an error
    fn execute(&self, env: &mut EnvironmentStack) -> Result<NodeOutcome, ShivaError>;
}
