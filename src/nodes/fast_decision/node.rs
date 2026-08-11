// Fast Decision Engine Node — Phase 2 of the 3-Phase Consensus Pipeline
//
// WHAT THIS FILE DOES:
// Implements the Fast Decision Engine, one of three Phase 2 consensus engines.
// Generates the baseline continuous motor proposal a_fast using the Soft Actor-Critic
// (SAC) policy via the PolicyEvaluator brain trait.
//
// HOW IT DOES IT:
// Holds a `Box<dyn PolicyEvaluator>` trait object from `src/brain/core/traits.rs`.
// Exposes a single public method `execute()` that reads current_state and prev_action
// from the EnvironmentStack, calls `evaluate_policy()`, and writes the result to
// `env.policy_output`.
//
// MATHEMATICAL CONTEXT:
// The SAC policy samples actions from a squashed Gaussian distribution:
//   a_fast = \tanh(\mu_\phi(S_t) + \sigma_\phi(S_t) \odot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)
// The confidence weight w_fast is computed from the action delta:
//   w_fast = \exp\left(-0.1 \sum_i |a_{fast,i} - a_{prev,i}|\right)
//
// WHY WE DO THIS:
// The Fast Decision Engine produces the primary motor command that forms the
// foundation of the Phase 2 consensus. Its confidence weight w_fast determines
// how strongly this proposal influences the merged candidate action.

use crate::brain::core::traits::PolicyEvaluator;
use crate::nodes::core::shared_state::EnvironmentStack;

/// FastDecisionNode — Phase 2 baseline motor policy engine.
///
/// WHAT: Generates the primary continuous motor proposal a_fast and confidence w_fast.
/// HOW: Delegates to a `Box<dyn PolicyEvaluator>` wrapping the SAC algorithm.
/// WHY: Provides the foundational action estimate for the Phase 2 weighted consensus.
pub struct FastDecisionNode {
    /// Trait object wrapping the SAC policy evaluation implementation.
    /// WHAT: The brain-layer adapter implementing policy evaluation.
    /// WHY: Dependency inversion — this node never imports from `crate::algorithms`.
    evaluator: Box<dyn PolicyEvaluator>,
}

impl FastDecisionNode {
    /// Creates a new FastDecisionNode with the given policy evaluator implementation.
    ///
    /// WHAT: Constructor accepting any `PolicyEvaluator` trait implementor.
    /// HOW: Wraps the evaluator in a `Box<dyn PolicyEvaluator>` for dynamic dispatch.
    /// WHY: Allows swapping policy strategies (e.g., SAC → PPO) without changing node code.
    pub fn new(evaluator: Box<dyn PolicyEvaluator>) -> Self {
        Self { evaluator }
    }

    /// Single public method: executes Phase 2 policy evaluation.
    ///
    /// WHAT: Computes the baseline motor proposal a_fast with confidence weight w_fast.
    ///
    /// HOW:
    /// 1. Reads `env.current_state` (S_t) and `env.prev_action` (a*_{t-1}).
    /// 2. Calls `evaluator.evaluate_policy(current_state, prev_action)` via the brain trait.
    /// 3. Writes the resulting `PolicyProposal` to `env.policy_output`.
    ///
    /// WHY: The orchestrator uses `env.policy_output.proposed_action` as a_fast and
    /// `env.policy_output.confidence_weight` as w_fast in the Phase 2 consensus formula.
    pub fn execute(&self, env: &mut EnvironmentStack) {
        // Evaluate the SAC policy via the brain trait interface.
        // Returns PolicyProposal { proposed_action: [f32; 32], confidence_weight, entropy_score }
        let proposal = self.evaluator.evaluate_policy(
            &env.current_state,
            &env.prev_action,
        );

        // Write result to the shared EnvironmentStack.
        env.policy_output = proposal;
    }
}
