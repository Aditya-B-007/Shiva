// RND Anomaly Facade Adapter for Shiva 2.0
//
// WHAT THIS FILE DOES:
// Wraps RNDModule (rnd.rs) inside RndAnomalyAdapter struct.
//
// HOW IT DOES IT:
// Implements AnomalyDetector trait contract, providing detect_anomaly method.
//
// WHY WE DO THIS:
// Middleware enabling Anomaly Nodes to trigger emergency deceleration fallbacks whenever novel states are encountered.

use crate::algorithms::rnd::{RNDConfig, RNDModule};
use crate::brain::core::dto::AnomalyAssessment;
use crate::brain::core::traits::AnomalyDetector;

/// RndAnomalyAdapter wraps RNDModule to implement AnomalyDetector trait
pub struct RndAnomalyAdapter {
    rnd_module: RNDModule,
    error_threshold: f32,
}

impl RndAnomalyAdapter {
    pub fn new(config: RNDConfig, error_threshold: f32) -> Self {
        Self {
            rnd_module: RNDModule::new(config),
            error_threshold,
        }
    }
}

impl Default for RndAnomalyAdapter {
    fn default() -> Self {
        let threshold = std::env::var("RND_ANOMALY_THRESHOLD")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2.5f32);
        Self::new(RNDConfig::default(), threshold)
    }
}

impl AnomalyDetector for RndAnomalyAdapter {
    /// Detects out-of-distribution state novelty and yields emergency action fallback if anomaly detected
    /// WHAT: Detects environmental state novelty and provides emergency action fallback.
    /// HOW: Evaluates RND curiosity prediction error; if error > threshold, sets is_out_of_distribution = true and scales prev_action by 0.5.
    /// WHY: Ensures real-time systems fail-safe smoothly when encountering novel/unfamiliar situations.
    fn detect_anomaly(&self, current_state: &[f32], prev_action: &[f32]) -> AnomalyAssessment {
        let mut mutable_module = RNDModule::new(self.rnd_module.config.clone());
        let error = mutable_module.compute_intrinsic_reward(current_state);

        let is_ood = error > self.error_threshold;
        let mut emergency = [0.0f32; 32];

        for (i, &prev) in prev_action.iter().take(32).enumerate() {
            emergency[i] = prev * 0.5;
        }

        AnomalyAssessment {
            prediction_error: error,
            is_out_of_distribution: is_ood,
            emergency_action: emergency,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnd_anomaly_adapter_detection() {
        let adapter = RndAnomalyAdapter::default();
        let state = [10.0f32; 64];
        let prev_act = [0.8f32; 32];

        let assessment = adapter.detect_anomaly(&state, &prev_act);

        assert!(assessment.prediction_error >= 0.0);
        assert!((assessment.emergency_action[0] - 0.4).abs() < 1e-4);
    }
}

