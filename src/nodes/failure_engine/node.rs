// Failure Engine Node — Phase 1 of the 3-Phase Consensus Pipeline
//
// WHAT THIS FILE DOES:
// Implements the Failure Engine (Phase 1) of the Mothership Ensemble.
// Before evaluating any standard control policy, the system checks whether
// the environment state S_t has drifted into completely unknown or catastrophic territory.
//
// HOW IT DOES IT:
// Holds a `Box<dyn AnomalyDetector>` trait object from `src/brain/core/traits.rs`.
// Exposes a single public method `execute()` that reads current_state and prev_action
// from the EnvironmentStack, calls `detect_anomaly()`, and writes the result to
// `env.anomaly_output`.
//
// MATHEMATICAL FORMULATION:
// The Failure Engine measures feature prediction error E(S_t) between fixed target
// and trained predictor networks:
//   E(S_t) = \frac{1}{k} \| \hat{f}_\theta(S_t) - f^*(S_t) \|_2^2
//
// PRECEDENCE RULE:
// If E(S_t) > threshold_{OOD}, the system bypasses all other policy engines and
// dispatches a pre-compiled safe emergency recovery action a_{emergency} directly
// to hardware.
//
// WHY WE DO THIS:
// In safety-critical real-time systems, encountering an unknown state must trigger
// immediate protective action BEFORE any policy computation runs. The Failure Engine
// acts as the first line of defense.

use crate::brain::core::traits::AnomalyDetector;
use crate::nodes::core::shared_state::EnvironmentStack;

/// FailureEngineNode — Phase 1 anomaly gate of the Mothership Ensemble.
///
/// WHAT: Detects out-of-distribution environment states and triggers emergency fallback.
/// HOW: Holds a `Box<dyn AnomalyDetector>` and delegates to its `detect_anomaly()` method.
/// WHY: Must execute FIRST in every cycle to protect against catastrophic unknowns.
pub struct FailureEngineNode {
    /// Trait object wrapping the RND anomaly detection implementation.
    /// WHAT: The brain-layer adapter implementing anomaly detection.
    /// WHY: Dependency inversion — this node never imports from `crate::algorithms`.
    detector: Box<dyn AnomalyDetector>,
}

impl FailureEngineNode {
    /// Creates a new FailureEngineNode with the given anomaly detector implementation.
    ///
    /// WHAT: Constructor accepting any `AnomalyDetector` trait implementor.
    /// HOW: Wraps the detector in a `Box<dyn AnomalyDetector>` for dynamic dispatch.
    /// WHY: Allows swapping anomaly detection strategies without changing node code.
    pub fn new(detector: Box<dyn AnomalyDetector>) -> Self {
        Self { detector }
    }

    /// Single public method: executes Phase 1 anomaly evaluation.
    ///
    /// WHAT: Checks if the current environment state S_t is out-of-distribution.
    ///
    /// HOW:
    /// 1. Reads `env.current_state` (sensor observation S_t) and `env.prev_action` (a*_{t-1}).
    /// 2. Calls `detector.detect_anomaly(current_state, prev_action)` via the brain trait.
    /// 3. Writes the resulting `AnomalyAssessment` to `env.anomaly_output`.
    /// 4. If `is_out_of_distribution == true`, sets `env.is_emergency = true`.
    ///
    /// WHY: The orchestrator checks `env.anomaly_output.is_out_of_distribution` after this
    /// call. If true, it short-circuits the pipeline and dispatches `emergency_action`
    /// directly, skipping Phases 2 and 3.
    pub fn execute(&self, env: &mut EnvironmentStack) {
        // Evaluate anomaly detection via the brain trait interface.
        // The detector computes E(S_t) = (1/k) * ||f_hat(S_t) - f*(S_t)||^2
        // and flags is_out_of_distribution if E(S_t) > threshold.
        let assessment = self.detector.detect_anomaly(
            &env.current_state,
            &env.prev_action,
        );

        // Write result to the shared EnvironmentStack.
        env.anomaly_output = assessment;

        // If the state is out-of-distribution, flag the emergency.
        // The orchestrator will read this and short-circuit to emergency_action.
        if assessment.is_out_of_distribution {
            env.is_emergency = true;
        }
    }
}
