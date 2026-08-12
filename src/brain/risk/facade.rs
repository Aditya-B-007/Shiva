// IQN Risk Facade Adapter for Shiva 2.0
//
// WHAT THIS FILE DOES:
// Wraps IQNAgent (implicitQuantileNetworks.rs) inside IqnRiskAdapter struct.
//
// HOW IT DOES IT:
// Implements RiskEvaluator trait contract, providing evaluate_risk method.
//
// WHY WE DO THIS:
// Middleware enabling Risk Evaluation Nodes to calculate tail risks (CVaR) without needing direct dependency on IQN internal structures.

use crate::algorithms::implicitQuantileNetworks::{IQNAgent, IQNConfig};
use crate::brain::core::dto::RiskAssessment;
use crate::brain::core::traits::RiskEvaluator;

/// IqnRiskAdapter wraps IQNAgent to implement RiskEvaluator trait
pub struct IqnRiskAdapter {
    agent: IQNAgent,
}

impl IqnRiskAdapter {
    pub fn new(config: IQNConfig, capacity: usize) -> Self {
        Self {
            agent: IQNAgent::new(config, capacity),
        }
    }
}

impl Default for IqnRiskAdapter {
    fn default() -> Self {
        Self::new(IQNConfig::default(), 10_000)
    }
}

impl RiskEvaluator for IqnRiskAdapter {
    /// Evaluates multi-step trajectory tail risk (CVaR) using IQN quantile distribution
    /// WHAT: Computes risk metrics over state history.
    /// HOW: Evaluates quantile return predictions across low alpha fractions (0.01 to 0.10) to calculate CVaR score.
    /// WHY: Quantifies worst-case tail risks to protect against dangerous multi-step trajectories.
    fn evaluate_risk(&self, state_history: &[f32], _action_history: &[f32]) -> RiskAssessment {
        let alpha_taus = [0.01f32, 0.02, 0.05, 0.08, 0.10];
        let quantiles = self.agent.online_net.evaluate_quantiles(state_history, &alpha_taus);

        let mut tail_sum = 0.0f32;
        let n_samples = alpha_taus.len() as f32;

        for q_vec in quantiles.iter() {
            if let Some(&min_q) = q_vec.iter().min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)) {
                tail_sum += min_q;
            }
        }

        let cvar_score = tail_sum / n_samples;
        let adjusted_weight = (1.0 + cvar_score * 0.1).clamp(0.0, 1.0);

        RiskAssessment {
            cvar_risk_score: cvar_score,
            risk_adjusted_weight: adjusted_weight,
        }
    }
}

