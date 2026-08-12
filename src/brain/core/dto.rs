// Data Transfer Objects for Shiva 2.0 Brain Layer
// All structures are C-aligned to 64 bytes (cache-line / SIMD friendly) with zero hot-path heap allocations.
//
// WHAT THIS FILE DOES:
// Defines lightweight data containers (DTOs) passed between brain node interfaces.
//
// HOW IT DOES IT:
// Uses #[repr(C, align(64))] and fixed-size stack arrays [f32; 32] / [u8; 32] instead of dynamic Heap Vectors.
//
// WHY WE DO THIS:
// Eliminates dynamic heap memory allocation (zero GC / zero malloc) during real-time control loops, achieving sub-millisecond execution.

/// Motor policy proposal output from SAC algorithm
/// WHAT: Data packet containing proposed motor control actions.
/// HOW: Stores a fixed 32-float action array along with confidence and entropy scores.
/// WHY: Allows the policy node to share motor proposals without heap allocations.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct PolicyProposal {
    pub proposed_action: [f32; 32],
    pub confidence_weight: f32,
    pub entropy_score: f32,
}

impl Default for PolicyProposal {
    fn default() -> Self {
        Self {
            proposed_action: [0.0; 32],
            confidence_weight: 1.0,
            entropy_score: 0.0,
        }
    }
}

/// Safety and rule evaluation result from CPO algorithm
/// WHAT: Result packet from physical constraint and rule checking.
/// HOW: Contains rule bitmasks, modified safe action proposals, and a boolean veto flag.
/// WHY: Allows safety nodes to overwrite dangerous actions or halt execution instantly.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct ConstraintResult {
    pub rule_mask: [u8; 32],
    pub projected_action: [f32; 32],
    pub is_vetoed: bool,
}

impl Default for ConstraintResult {
    fn default() -> Self {
        Self {
            rule_mask: [0; 32],
            projected_action: [0.0; 32],
            is_vetoed: false,
        }
    }
}

/// Multi-step tail risk assessment from IQN algorithm
/// WHAT: Quantile risk metrics evaluating trajectory tail risks.
/// HOW: Computes Conditional Value-at-Risk (CVaR) score and risk-adjusted weight.
/// WHY: Enables decision nodes to penalize high-variance or risky trajectory options.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct RiskAssessment {
    pub cvar_risk_score: f32,
    pub risk_adjusted_weight: f32,
}

impl Default for RiskAssessment {
    fn default() -> Self {
        Self {
            cvar_risk_score: 0.0,
            risk_adjusted_weight: 1.0,
        }
    }
}

/// Skill-conditioned adaptation proposal from TD3 algorithm
/// WHAT: Proposal for executing specific sub-skills or behaviors.
/// HOW: Carries sub-skill action array, adaptation confidence weight, and 32-byte skill identifier.
/// WHY: Facilitates hierarchical control by letting specialized skill controllers override default motor actions.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct AdaptedProposal {
    pub adapted_action: [f32; 32],
    pub adaptation_weight: f32,
    pub active_skill_id: [u8; 32],
}

impl Default for AdaptedProposal {
    fn default() -> Self {
        Self {
            adapted_action: [0.0; 32],
            adaptation_weight: 1.0,
            active_skill_id: [0; 32],
        }
    }
}

/// Out-of-distribution anomaly assessment from RND algorithm
/// WHAT: Anomaly metrics evaluating novelty and triggering emergency fallbacks.
/// HOW: Measures prediction error and provides an emergency deceleration action array.
/// WHY: Ensures the system can enter a safe emergency fallback state if an unfamiliar environment state occurs.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct AnomalyAssessment {
    pub prediction_error: f32,
    pub is_out_of_distribution: bool,
    pub emergency_action: [f32; 32],
}

impl Default for AnomalyAssessment {
    fn default() -> Self {
        Self {
            prediction_error: 0.0,
            is_out_of_distribution: false,
            emergency_action: [0.0; 32],
        }
    }
}
