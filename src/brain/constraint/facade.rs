// CPO Constraint Facade Adapter for Shiva 2.0
//
// WHAT THIS FILE DOES:
// Wraps Constrained Policy Optimizer (cpo.rs) inside CpoConstraintAdapter struct.
//
// HOW IT DOES IT:
// Implements ConstraintEvaluator trait contract, providing evaluate_constraints method.
//
// WHY WE DO THIS:
// Middleware enabling Safety Nodes to enforce physical rate-of-change and safety bounds without tight coupling to CPO implementation.

use crate::algorithms::cpo::{CPOConfig, ConstrainedPolicyOptimizer};
use crate::brain::core::dto::ConstraintResult;
use crate::brain::core::traits::ConstraintEvaluator;

/// CpoConstraintAdapter wraps ConstrainedPolicyOptimizer to implement ConstraintEvaluator
pub struct CpoConstraintAdapter {
    optimizer: ConstrainedPolicyOptimizer,
}

impl CpoConstraintAdapter {
    pub fn new(config: CPOConfig, buffer_capacity: usize) -> Self {
        Self {
            optimizer: ConstrainedPolicyOptimizer::new(config, buffer_capacity),
        }
    }
}

impl Default for CpoConstraintAdapter {
    fn default() -> Self {
        Self::new(CPOConfig::default(), 10_000)
    }
}

impl ConstraintEvaluator for CpoConstraintAdapter {
    /// Evaluates rate-of-change delta, boundary limits, and rules using zero dynamic heap allocations
    /// WHAT: Evaluates safety boundaries and rate-of-change limits.
    /// HOW: Clamps max_delta between prev_action and proposed_action, checking rule_flags bitmask for veto conditions.
    /// WHY: Ensures physical control actions never exceed maximum safe rate of change.
    fn evaluate_constraints(
        &self,
        proposed_action: &[f32],
        prev_action: &[f32],
        rule_flags: &[u8],
    ) -> ConstraintResult {
        let mut projected = [0.0f32; 32];
        let mut mask = [0u8; 32];
        let mut vetoed = false;

        for (i, &flag) in rule_flags.iter().take(32).enumerate() {
            mask[i] = flag;
        }

        let max_delta = 0.35f32;

        for i in 0..32 {
            let prop = proposed_action.get(i).copied().unwrap_or(0.0);
            let prev = prev_action.get(i).copied().unwrap_or(0.0);

            let delta = prop - prev;
            let clamped_delta = delta.clamp(-max_delta, max_delta);
            let constrained_val = (prev + clamped_delta).clamp(-1.0, 1.0);

            projected[i] = constrained_val;

            if mask[i] == 1 && prop.abs() > 0.05 {
                vetoed = true;
            }
        }

        ConstraintResult {
            rule_mask: mask,
            projected_action: projected,
            is_vetoed: vetoed,
        }
    }


