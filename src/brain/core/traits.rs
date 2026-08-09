// Brain Layer Trait Contracts for Shiva 2.0
//
// WHAT THIS FILE DOES:
// Defines the 5 core Rust trait contracts for Mothership Node interfaces.
//
// HOW IT DOES IT:
// Requires thread-safe bounds (Send + Sync) and standardized method signatures for each domain task.
//
// WHY WE DO THIS:
// Enforces the Dependency Inversion Principle so node execution domains never depend directly on underlying mathematical algorithms.

use crate::brain::core::dto::*;

/// Policy Evaluator Trait Contract
/// WHAT: Interface for motor policy evaluation.
/// HOW: Evaluates current state and previous action to return a PolicyProposal.
/// WHY: Decouples motor nodes from specific RL policy implementations (like SAC).
pub trait PolicyEvaluator: Send + Sync {
    /// Evaluates primary motor proposal (SAC)
    fn evaluate_policy(&self, state_slice: &[f32], prev_action: &[f32]) -> PolicyProposal;
}

/// Constraint Evaluator Trait Contract
/// WHAT: Interface for physical safety rule evaluation.
/// HOW: Evaluates proposed actions against prev_action and rule_flags.
/// WHY: Decouples safety nodes from specific optimization algorithms (like CPO).
pub trait ConstraintEvaluator: Send + Sync {
    /// Evaluates physical rules, rate-of-change, and safety bounds (CPO)
    fn evaluate_constraints(
        &self, 
        proposed_action: &[f32], 
        prev_action: &[f32], 
        rule_flags: &[u8]
    ) -> ConstraintResult;
}

/// Risk Evaluator Trait Contract
/// WHAT: Interface for multi-step trajectory risk estimation.
/// HOW: Evaluates state and action history to output a RiskAssessment.
/// WHY: Decouples risk nodes from specific distributional RL estimators (like IQN).
pub trait RiskEvaluator: Send + Sync {
    /// Evaluates multi-step trajectory tail-risk (IQN / CVaR)
    fn evaluate_risk(&self, state_history: &[f32], action_history: &[f32]) -> RiskAssessment;
}

/// Adaptation Evaluator Trait Contract
/// WHAT: Interface for skill option execution.
/// HOW: Evaluates current state and skill_id to produce an AdaptedProposal.
/// WHY: Decouples skill nodes from specific sub-skill models (like TD3 + z).
pub trait AdaptationEvaluator: Send + Sync {
    /// Evaluates drift compensation & ONNX skill option execution (TD3 + z)
    fn evaluate_adaptation(
        &self, 
        current_state: &[f32], 
        prev_action: &[f32], 
        skill_id: &[u8; 32]
    ) -> AdaptedProposal;
}

/// Anomaly Detector Trait Contract
/// WHAT: Interface for novelty detection and emergency fallback.
/// HOW: Evaluates state novelty to output an AnomalyAssessment.
/// WHY: Decouples anomaly monitoring from specific curiosity modules (like RND).
pub trait AnomalyDetector: Send + Sync {
    /// Evaluates out-of-distribution state novelty and triggers emergency fallback (RND)
    fn detect_anomaly(&self, current_state: &[f32], prev_action: &[f32]) -> AnomalyAssessment;
}

