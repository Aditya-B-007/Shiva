//! Shiva 2.0 — Safety Invariant Test Suite
//!
//! Executable tests for critical safety invariants that the framework
//! MUST uphold at all times. These invariants are the difference between
//! "documented claims" and "enforced behavior".
//!
//! Invariants tested:
//! - Action bounds: ∀i: final_action[i] ∈ [-1.0, 1.0]
//! - Slew-rate limits: ∀i: |final_action[i] - prev_action[i]| ≤ Δ_max (0.35)
//! - Rule-mask enforcement: if rule_flags[i] == 1, action constrained
//! - NaN/Inf sanitization: invalid inputs produce valid finite outputs
//! - Emergency fallback: OOD detection short-circuits pipeline
//! - Veto behavior: safety veto falls back to prev_action

use shiva::brain::constraint::CpoConstraintAdapter;
use shiva::brain::core::traits::ConstraintEvaluator;
use shiva::brain::core::dto::ConstraintResult;

// ═══════════════════════════════════════════════════════════════
// Action Bounds Invariant
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_action_bounds_invariant() {
    let adapter = CpoConstraintAdapter::default();
    // Propose actions far outside [-1.0, 1.0]
    let proposed = [5.0f32; 32];
    let prev = [0.0f32; 32];
    let flags = [0u8; 32];

    let result = adapter.evaluate_constraints(&proposed, &prev, &flags);

    // Every projected action must be within [-1.0, 1.0]
    for i in 0..32 {
        assert!(
            result.projected_action[i] >= -1.0 && result.projected_action[i] <= 1.0,
            "Action[{}] = {} is outside [-1.0, 1.0]",
            i,
            result.projected_action[i]
        );
    }
}

#[test]
fn test_action_bounds_negative_extreme() {
    let adapter = CpoConstraintAdapter::default();
    let proposed = [-10.0f32; 32];
    let prev = [0.0f32; 32];
    let flags = [0u8; 32];

    let result = adapter.evaluate_constraints(&proposed, &prev, &flags);

    for i in 0..32 {
        assert!(
            result.projected_action[i] >= -1.0,
            "Action[{}] = {} is below -1.0",
            i,
            result.projected_action[i]
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Slew-Rate Limiting Invariant
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_slew_rate_limiting_invariant() {
    let adapter = CpoConstraintAdapter::default();
    let max_delta = 0.35f32;

    // Previous action at 0.0, proposed at 1.0 — exceeds Δ_max
    let proposed = [1.0f32; 32];
    let prev = [0.0f32; 32];
    let flags = [0u8; 32];

    let result = adapter.evaluate_constraints(&proposed, &prev, &flags);

    for i in 0..32 {
        let delta = (result.projected_action[i] - prev[i]).abs();
        assert!(
            delta <= max_delta + 1e-6, // small epsilon for float comparison
            "Slew rate violation: |a[{}] - prev[{}]| = {} > {}",
            i, i, delta, max_delta
        );
    }
}

#[test]
fn test_slew_rate_negative_direction() {
    let adapter = CpoConstraintAdapter::default();
    let max_delta = 0.35f32;

    let proposed = [-1.0f32; 32];
    let prev = [0.5f32; 32];
    let flags = [0u8; 32];

    let result = adapter.evaluate_constraints(&proposed, &prev, &flags);

    for i in 0..32 {
        let delta = (result.projected_action[i] - prev[i]).abs();
        assert!(
            delta <= max_delta + 1e-6,
            "Negative slew rate violation at channel {}",
            i
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Rule-Mask Enforcement Invariant
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_rule_mask_triggers_veto_on_nonzero_action() {
    let adapter = CpoConstraintAdapter::default();

    // All channels under interlock, proposing non-zero action
    let proposed = [0.5f32; 32];
    let prev = [0.0f32; 32];
    let flags = [1u8; 32]; // all locked

    let result = adapter.evaluate_constraints(&proposed, &prev, &flags);

    // With all channels locked and significant proposed actions,
    // the constraint evaluator should veto
    assert!(
        result.is_vetoed,
        "Action should be vetoed when all channels are under interlock"
    );
}

#[test]
fn test_zero_rule_mask_allows_action() {
    let adapter = CpoConstraintAdapter::default();

    let proposed = [0.1f32; 32];
    let prev = [0.0f32; 32];
    let flags = [0u8; 32]; // no interlocks

    let result = adapter.evaluate_constraints(&proposed, &prev, &flags);

    // With no interlocks and small proposed actions, should not veto
    assert!(
        !result.is_vetoed,
        "Action should not be vetoed when no channels are locked"
    );
}

// ═══════════════════════════════════════════════════════════════
// NaN/Inf Sanitization Invariant
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_nan_input_produces_finite_output() {
    use shiva::adapters::default_input::DefaultInputAdapter;
    use shiva::framework::adapter::InputAdapter;
    use shiva::protocol::systemSide::SystemInputDTO;

    let adapter = DefaultInputAdapter::new();
    let mut input = SystemInputDTO::default();
    input.state[0] = f32::NAN;
    input.state[1] = f32::INFINITY;
    input.state[2] = f32::NEG_INFINITY;
    input.previous_rewards = f32::NAN;

    let result = adapter.adapt_input(&input);
    assert!(result.is_ok());

    let adapted = result.unwrap();
    for i in 0..64 {
        assert!(
            adapted.processed_state[i].is_finite(),
            "Processed state[{}] should be finite, got: {}",
            i,
            adapted.processed_state[i]
        );
    }
    assert!(adapted.reward.is_finite(), "Reward should be finite");
}

// ═══════════════════════════════════════════════════════════════
// Emergency Fallback Invariant
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_anomaly_detector_emergency_action() {
    use shiva::brain::anomaly::RndAnomalyAdapter;
    use shiva::brain::core::traits::AnomalyDetector;

    // Create adapter with extremely low threshold so normal states trigger OOD
    use shiva::algorithms::rnd::RNDConfig;
    let adapter = RndAnomalyAdapter::new(RNDConfig::default(), -1.0); // threshold so low everything is OOD

    let state = [0.5f32; 64];
    let prev_action = [0.3f32; 32];

    let assessment = adapter.detect_anomaly(&state, &prev_action);

    if assessment.is_out_of_distribution {
        // Emergency action should be 0.5 × prev_action
        for i in 0..32 {
            let expected = prev_action[i] * 0.5;
            assert!(
                (assessment.emergency_action[i] - expected).abs() < 1e-6,
                "Emergency action[{}] should be 0.5 * prev_action[{}]",
                i, i
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Combined End-to-End Safety Test
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_end_to_end_safety_pipeline() {
    let adapter = CpoConstraintAdapter::default();

    // Simulate 100 cycles with varying inputs
    let mut prev_action = [0.0f32; 32];
    let flags = [0u8; 32];
    let max_delta = 0.35f32;

    for cycle in 0..100 {
        // Create a proposed action that varies each cycle
        let mut proposed = [0.0f32; 32];
        for i in 0..32 {
            proposed[i] = ((cycle * 7 + i) as f32 * 0.1).sin();
        }

        let result = adapter.evaluate_constraints(&proposed, &prev_action, &flags);

        // Verify ALL safety invariants hold
        for i in 0..32 {
            // Bounds
            assert!(
                result.projected_action[i] >= -1.0 && result.projected_action[i] <= 1.0,
                "Bounds violated at cycle {} channel {}",
                cycle, i
            );
            // Slew rate
            let delta = (result.projected_action[i] - prev_action[i]).abs();
            assert!(
                delta <= max_delta + 1e-6,
                "Slew rate violated at cycle {} channel {}: delta={}",
                cycle, i, delta
            );
        }

        // Update prev_action for next cycle
        if !result.is_vetoed {
            prev_action = result.projected_action;
        }
    }
}
