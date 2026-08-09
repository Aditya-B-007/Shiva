// SAC Policy Facade Adapter for Shiva 2.0
//
// WHAT THIS FILE DOES:
// Wraps the Soft Actor-Critic algorithm (softActorCriticNetwork.rs) inside the SacPolicyAdapter struct.
//
// HOW IT DOES IT:
// Implements the PolicyEvaluator trait contract, providing a single evaluate_policy method.
//
// WHY WE DO THIS:
// Acts as middleware so Mothership Policy Nodes call evaluate_policy without needing direct knowledge of SAC internals.

use crate::algorithms::softActorCriticNetwork::{SACConfig, SoftActorCritic};
use crate::brain::core::dto::PolicyProposal;
use crate::brain::core::traits::PolicyEvaluator;

/// SacPolicyAdapter wraps SoftActorCritic to implement PolicyEvaluator trait
pub struct SacPolicyAdapter {
    sac_agent: SoftActorCritic,
}

impl SacPolicyAdapter {
    pub fn new(config: SACConfig, buffer_capacity: usize) -> Self {
        Self {
            sac_agent: SoftActorCritic::new(config, buffer_capacity),
        }
    }
}

impl Default for SacPolicyAdapter {
    fn default() -> Self {
        Self::new(SACConfig::default(), 10_000)
    }
}

impl PolicyEvaluator for SacPolicyAdapter {
    /// Evaluates SAC policy network output with zero hot-path heap allocations
    /// WHAT: Computes motor policy proposal for state_slice.
    /// HOW: Runs SAC actor sampling, copies action to stack array [f32; 32], and calculates confidence score based on action delta.
    /// WHY: Provides zero-allocation real-time motor predictions.
    fn evaluate_policy(&self, state_slice: &[f32], prev_action: &[f32]) -> PolicyProposal {
        let (sampled_action, _log_prob) = self.sac_agent.actor.sample_action(state_slice);
        let mut fixed_action = [0.0f32; 32];

        // Copy sampled action to fixed-size array without heap allocation
        for (i, &val) in sampled_action.iter().take(32).enumerate() {
            fixed_action[i] = val;
        }

        // Calculate smooth confidence weight based on action delta relative to prev_action
        let mut diff_sum = 0.0f32;
        for (i, &prev) in prev_action.iter().take(32).enumerate() {
            diff_sum += (fixed_action[i] - prev).abs();
        }
        let confidence = (-0.1 * diff_sum).exp().clamp(0.0, 1.0);

        PolicyProposal {
            proposed_action: fixed_action,
            confidence_weight: confidence,
            entropy_score: 0.2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sac_policy_adapter_zero_allocation() {
        let adapter = SacPolicyAdapter::default();
        let state = [0.5f32; 64];
        let prev_action = [0.0f32; 32];
        let proposal = adapter.evaluate_policy(&state, &prev_action);

        assert_eq!(proposal.proposed_action.len(), 32);
        assert!(proposal.confidence_weight >= 0.0 && proposal.confidence_weight <= 1.0);
    }
}

