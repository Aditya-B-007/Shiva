// Explorer Engine Node — Phase 2 of the 3-Phase Consensus Pipeline
//
// WHAT THIS FILE DOES:
// Implements the Explorer Engine, one of three Phase 2 consensus engines.
// Detects physical drift, friction changes, or external turbulence and contributes
// a drift-compensated skill action a_explore with adaptation weight w_adapt.
//
// HOW IT DOES IT:
// Holds a `Box<dyn AdaptationEvaluator>` trait object from `src/brain/core/traits.rs`.
// Exposes a single public method `execute()` that reads current_state, prev_action,
// and active_skill_id from the EnvironmentStack, calls `evaluate_adaptation()`,
// and writes the result to `env.adaptation_output`.
//
// MATHEMATICAL CONTEXT:
// The Explorer Engine uses TD3 with latent skill embedding z:
//   a_explore = \pi_\phi(S_t, z), \quad z = \text{decode}(\text{skill\_id})
// where z \in \mathbb{R}^{16} is the latent skill vector decoded from the 32-byte
// skill identifier. The decode maps each byte to [-1.0, 1.0]:
//   z_i = \frac{skill\_id[i]}{255} \cdot 2 - 1
//
// WHY WE DO THIS:
// Standard motor policies assume stationary dynamics. In real-world operation,
// physical parameters (friction, mass, wind) drift over time. The Explorer Engine
// activates specialized sub-skills conditioned on latent vector z to compensate
// for these dynamic changes without retraining the base policy.

use crate::brain::core::traits::AdaptationEvaluator;
use crate::framework::node::{Node, Phase, NodeOutcome};
use crate::framework::error::ShivaError;
use crate::nodes::core::shared_state::EnvironmentStack;

/// ExplorerNode — Phase 2 drift compensation and skill adaptation engine.
///
/// WHAT: Produces drift-compensated skill action a_explore with adaptation weight w_adapt.
/// HOW: Delegates to a `Box<dyn AdaptationEvaluator>` wrapping the TD3 + z algorithm.
/// WHY: Compensates for physical dynamics drift using hierarchical skill-conditioned control.
pub struct ExplorerNode {
    /// Trait object wrapping the TD3 + z adaptation evaluation implementation.
    /// WHAT: The brain-layer adapter implementing skill adaptation.
    /// WHY: Dependency inversion — this node never imports from `crate::algorithms`.
    evaluator: Box<dyn AdaptationEvaluator>,
}

impl ExplorerNode {
    /// Creates a new ExplorerNode with the given adaptation evaluator implementation.
    ///
    /// WHAT: Constructor accepting any `AdaptationEvaluator` trait implementor.
    /// HOW: Wraps the evaluator in a `Box<dyn AdaptationEvaluator>` for dynamic dispatch.
    /// WHY: Allows swapping skill adaptation strategies without changing node code.
    pub fn new(evaluator: Box<dyn AdaptationEvaluator>) -> Self {
        Self { evaluator }
    }

    /// Single public method: executes Phase 2 skill adaptation.
    ///
    /// WHAT: Computes drift-compensated skill action a_explore with adaptation weight w_adapt.
    ///
    /// HOW:
    /// 1. Reads `env.current_state` (S_t), `env.prev_action` (a*_{t-1}), and
    ///    `env.active_skill_id` (32-byte skill embedding identifier).
    /// 2. Calls `evaluator.evaluate_adaptation(current_state, prev_action, skill_id)`
    ///    via the brain trait.
    /// 3. Writes the resulting `AdaptedProposal` to `env.adaptation_output`.
    ///
    /// WHY: The orchestrator uses `env.adaptation_output.adapted_action` as a_explore
    /// and `env.adaptation_output.adaptation_weight` as w_adapt in the Phase 2
    /// consensus formula.
    pub fn execute(&self, env: &mut EnvironmentStack) {
        // Evaluate skill-conditioned adaptation via the brain trait interface.
        // The evaluator decodes skill_id → z ∈ R^16 and computes a_explore = π(S_t, z).
        let proposal = self.evaluator.evaluate_adaptation(
            &env.current_state,
            &env.prev_action,
            &env.active_skill_id,
        );

        // Write result to the shared EnvironmentStack.
        env.adaptation_output = proposal;
    }
}

/// Node trait implementation for ExplorerNode.
impl Node for ExplorerNode {
    fn name(&self) -> &str {
        "Explorer"
    }

    fn phase(&self) -> Phase {
        Phase::Consensus
    }

    fn execute(&self, env: &mut EnvironmentStack) -> Result<NodeOutcome, ShivaError> {
        let proposal = self.evaluator.evaluate_adaptation(
            &env.current_state,
            &env.prev_action,
            &env.active_skill_id,
        );
        env.adaptation_output = proposal;
        Ok(NodeOutcome::Continue)
    }
}
