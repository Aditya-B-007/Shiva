// TD3 Skill Facade Adapter for Shiva 2.0
//
// WHAT THIS FILE DOES:
// Wraps TD3SkillAgent (td3.rs) inside Td3SkillAdapter struct.
//
// HOW IT DOES IT:
// Implements AdaptationEvaluator trait contract, providing evaluate_adaptation method.
//
// WHY WE DO THIS:
// Middleware enabling Skill Vault Nodes to execute skill options conditioned on latent skill vector z without direct coupling to TD3 algorithms.

use crate::algorithms::td3::{TD3Config, TD3SkillAgent};
use crate::brain::core::dto::AdaptedProposal;
use crate::brain::core::traits::AdaptationEvaluator;

/// Td3SkillAdapter wraps TD3SkillAgent to implement AdaptationEvaluator trait
pub struct Td3SkillAdapter {
    agent: TD3SkillAgent,
}

impl Td3SkillAdapter {
    pub fn new(config: TD3Config, capacity: usize) -> Self {
        Self {
            agent: TD3SkillAgent::new(config, capacity),
        }
    }
}

impl Default for Td3SkillAdapter {
    fn default() -> Self {
        Self::new(TD3Config::default(), 10_000)
    }
}

impl AdaptationEvaluator for Td3SkillAdapter {
    /// Evaluates skill option execution conditioned on latent skill vector z with zero heap allocations
    /// WHAT: Computes skill-conditioned adaptation proposal.
    /// HOW: Converts 32-byte skill_id into 16-float latent vector z, then invokes TD3 actor prediction.
    /// WHY: Facilitates hierarchical skill execution and drift compensation.
    fn evaluate_adaptation(
        &self,
        current_state: &[f32],
        _prev_action: &[f32],
        skill_id: &[u8; 32],
    ) -> AdaptedProposal {
        let mut skill_z = [0.0f32; 16];
        for (i, &b) in skill_id.iter().take(16).enumerate() {
            skill_z[i] = (b as f32 / 255.0) * 2.0 - 1.0;
        }

        let raw_action = self.agent.actor.predict_action(current_state, &skill_z);
        let mut adapted = [0.0f32; 32];

        for (i, &val) in raw_action.iter().take(32).enumerate() {
            adapted[i] = val;
        }

        AdaptedProposal {
            adapted_action: adapted,
            adaptation_weight: 0.85,
            active_skill_id: *skill_id,
        }
    }
}


