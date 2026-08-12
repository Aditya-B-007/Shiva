// Long Vision Engine Node — Phase 2 of the 3-Phase Consensus Pipeline
//
// WHAT THIS FILE DOES:
// Implements the Long Vision Engine, one of three Phase 2 consensus engines.
// Evaluates trajectory tail-risk (CVaR_alpha) across temporal history to produce
// a risk-adjusted weighting factor w_risk that modulates the consensus.
//
// HOW IT DOES IT:
// Holds a `Box<dyn RiskEvaluator>` trait object from `src/brain/core/traits.rs`.
// Exposes a single public method `execute()` that reads state_history and action_history
// from the EnvironmentStack, calls `evaluate_risk()`, and writes the result to
// `env.risk_output`.
//
// MATHEMATICAL FORMULATION:
// The Long Vision Engine evaluates the Conditional Value-at-Risk (CVaR) over
// the lower tail quantiles of the return distribution:
//   CVaR_\alpha = \frac{1}{\alpha} \int_0^\alpha F^{-1}_{Z}(\tau) \, d\tau
// where F^{-1}_Z is the quantile function of the return distribution Z(s, a).
//
// The risk-adjusted weight is:
//   w_risk = \text{clamp}(1.0 + CVaR \cdot 0.1, \, 0.0, \, 1.0)
//
// WHY WE DO THIS:
// In high-stakes real-time control, knowing only the average expected outcome is
// insufficient. The Long Vision Engine quantifies worst-case tail risks so the
// consensus can down-weight high-risk proposals or amplify risk-averse trajectories.
// This prevents the system from confidently executing actions that look good on
// average but have catastrophic failure modes.

use crate::brain::core::traits::RiskEvaluator;
use crate::nodes::core::shared_state::EnvironmentStack;

/// LongVisionNode — Phase 2 trajectory tail-risk evaluator.
///
/// WHAT: Evaluates multi-step trajectory risk and produces risk-adjusted weight w_risk.
/// HOW: Delegates to a `Box<dyn RiskEvaluator>` wrapping the IQN algorithm.
/// WHY: Modulates the Phase 2 consensus to penalize high-risk trajectory options.
pub struct LongVisionNode {
    /// Trait object wrapping the IQN risk evaluation implementation.
    /// WHAT: The brain-layer adapter implementing risk evaluation.
    /// WHY: Dependency inversion — this node never imports from `crate::algorithms`.
    evaluator: Box<dyn RiskEvaluator>,
}

impl LongVisionNode {
    /// Creates a new LongVisionNode with the given risk evaluator implementation.
    ///
    /// WHAT: Constructor accepting any `RiskEvaluator` trait implementor.
    /// HOW: Wraps the evaluator in a `Box<dyn RiskEvaluator>` for dynamic dispatch.
    /// WHY: Allows swapping risk estimation strategies without changing node code.
    pub fn new(evaluator: Box<dyn RiskEvaluator>) -> Self {
        Self { evaluator }
    }

    /// Single public method: executes Phase 2 risk evaluation.
    ///
    /// WHAT: Evaluates trajectory tail-risk (CVaR) and produces risk-adjusted weight.
    ///
    /// HOW:
    /// 1. Reads `env.state_history` and `env.action_history` (temporal windows).
    /// 2. Calls `evaluator.evaluate_risk(state_history, action_history)` via the brain trait.
    /// 3. Writes the resulting `RiskAssessment` to `env.risk_output`.
    ///
    /// WHY: The orchestrator uses `env.risk_output.risk_adjusted_weight` as w_risk
    /// in the Phase 2 normalized weighted consensus formula.
    pub fn execute(&self, env: &mut EnvironmentStack) {
        // Evaluate trajectory tail-risk via the brain trait interface.
        // The evaluator computes CVaR over low-alpha quantile fractions
        // and returns RiskAssessment { cvar_risk_score, risk_adjusted_weight }.
        let assessment = self.evaluator.evaluate_risk(
            &env.state_history,
            &env.action_history,
        );

        // Write result to the shared EnvironmentStack.
        env.risk_output = assessment;
    }
}
