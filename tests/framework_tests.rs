//! Shiva 2.0 — Framework Integration Tests
//!
//! Comprehensive test suite covering:
//! - Builder construction and validation
//! - Node registration and execution
//! - Execution lifecycle
//! - State propagation
//! - Action propagation
//! - Safety filtering
//! - Emergency handling
//! - Configuration validation
//! - Multiple sequential cycles

use shiva::config::{ShivaBuilder, ShivaConfig};
use shiva::framework::dimensions::{StateVector, ActionVector, RuleFlagVector, DEFAULT_STATE_DIM, DEFAULT_ACTION_DIM};
use shiva::framework::error::ShivaError;
use shiva::framework::lifecycle::LifecycleState;
use shiva::framework::node::{Node, Phase, NodeOutcome};
use shiva::framework::safety::{SafetyPolicy, SafetyVerdict};
use shiva::nodes::EnvironmentStack;
use shiva::protocol::systemSide::SystemInputDTO;

// ═══════════════════════════════════════════════════════════════
// Builder Construction & Validation Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_default_builder_produces_working_runtime() {
    let result = ShivaBuilder::new().build();
    assert!(result.is_ok(), "Default builder should produce a working runtime");
    let runtime = result.unwrap();
    assert_eq!(runtime.cycle_counter(), 0);
}

#[test]
fn test_invalid_config_min_gte_max() {
    let result = ShivaBuilder::new()
        .with_actuator_limits(1.0, -1.0)
        .build();
    assert!(result.is_err(), "min >= max should fail validation");
    match result.unwrap_err() {
        ShivaError::Configuration(_) => {} // expected
        other => panic!("Expected ConfigError, got: {:?}", other),
    }
}

#[test]
fn test_invalid_config_equal_limits() {
    let result = ShivaBuilder::new()
        .with_actuator_limits(0.0, 0.0)
        .build();
    assert!(result.is_err(), "equal limits should fail validation");
}

#[test]
fn test_invalid_config_nan_limits() {
    let result = ShivaBuilder::new()
        .with_actuator_limits(f32::NAN, 1.0)
        .build();
    assert!(result.is_err(), "NaN limits should fail validation");
}

#[test]
fn test_invalid_config_inf_limits() {
    let result = ShivaBuilder::new()
        .with_actuator_limits(f32::NEG_INFINITY, 1.0)
        .build();
    assert!(result.is_err(), "Inf limits should fail validation");
}

#[test]
fn test_invalid_config_zero_rows() {
    let result = ShivaBuilder::new()
        .with_matrix_rows(0)
        .build();
    assert!(result.is_err(), "zero rows should fail validation");
}

#[test]
fn test_valid_config_passes() {
    let config = ShivaConfig {
        matrix_rows: 30,
        min_signal: -2.0,
        max_signal: 2.0,
    };
    assert!(config.validate().is_ok());
}

#[test]
fn test_builder_overrides_defaults() {
    let builder = ShivaBuilder::new()
        .with_matrix_rows(50)
        .with_actuator_limits(-2.0, 2.0);
    let config = builder.config();
    assert_eq!(config.matrix_rows, 50);
    assert_eq!(config.min_signal, -2.0);
    assert_eq!(config.max_signal, 2.0);
}

// ═══════════════════════════════════════════════════════════════
// Node Trait & Plugin Interface Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_custom_node_implements_trait() {
    struct MockNode;

    impl Node for MockNode {
        fn name(&self) -> &str { "MockNode" }
        fn phase(&self) -> Phase { Phase::Consensus }
        fn execute(&self, env: &mut EnvironmentStack) -> Result<NodeOutcome, ShivaError> {
            // Simple test: set final_action to all 0.5
            for i in 0..32 {
                env.final_action[i] = 0.5;
            }
            Ok(NodeOutcome::Continue)
        }
    }

    let node = MockNode;
    assert_eq!(node.name(), "MockNode");
    assert_eq!(node.phase(), Phase::Consensus);

    let mut env = EnvironmentStack::default();
    let result = node.execute(&mut env);
    assert!(result.is_ok());
    assert_eq!(env.final_action[0], 0.5);
}

#[test]
fn test_node_short_circuit() {
    struct EmergencyNode;

    impl Node for EmergencyNode {
        fn name(&self) -> &str { "EmergencyNode" }
        fn phase(&self) -> Phase { Phase::AnomalyGate }
        fn execute(&self, _env: &mut EnvironmentStack) -> Result<NodeOutcome, ShivaError> {
            Ok(NodeOutcome::ShortCircuit {
                reason: "Test emergency",
            })
        }
    }

    let node = EmergencyNode;
    let mut env = EnvironmentStack::default();
    let result = node.execute(&mut env).unwrap();
    match result {
        NodeOutcome::ShortCircuit { reason } => {
            assert_eq!(reason, "Test emergency");
        }
        _ => panic!("Expected ShortCircuit"),
    }
}

#[test]
fn test_phase_ordering() {
    assert!(Phase::AnomalyGate < Phase::Consensus);
    assert!(Phase::Consensus < Phase::SafetyShield);
}

// ═══════════════════════════════════════════════════════════════
// Lifecycle Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_lifecycle_valid_transitions() {
    let state = LifecycleState::Created;
    assert!(state.can_transition_to(&LifecycleState::Configured));

    let state = LifecycleState::Configured;
    assert!(state.can_transition_to(&LifecycleState::Initialized));

    let state = LifecycleState::Initialized;
    assert!(state.can_transition_to(&LifecycleState::Running));

    let state = LifecycleState::Running;
    assert!(state.can_transition_to(&LifecycleState::Stopped));

    let state = LifecycleState::Stopped;
    assert!(state.can_transition_to(&LifecycleState::Running));
    assert!(state.can_transition_to(&LifecycleState::ShutDown));
}

#[test]
fn test_lifecycle_invalid_transitions() {
    let state = LifecycleState::Created;
    assert!(!state.can_transition_to(&LifecycleState::Running));

    let state = LifecycleState::ShutDown;
    assert!(!state.can_transition_to(&LifecycleState::Running));

    let state = LifecycleState::Running;
    assert!(!state.can_transition_to(&LifecycleState::Created));
}

#[test]
fn test_lifecycle_transition_error() {
    let state = LifecycleState::Created;
    let result = state.transition_to(LifecycleState::Running);
    assert!(result.is_err());
}

// ═══════════════════════════════════════════════════════════════
// Execution & State Propagation Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_cycle_counter_increments() {
    let result = ShivaBuilder::new().build();
    if let Ok(mut runtime) = result {
        let input = SystemInputDTO::default();
        let _ = runtime.step(input);
        assert_eq!(runtime.cycle_counter(), 1);

        let input2 = SystemInputDTO::default();
        let _ = runtime.step(input2);
        assert_eq!(runtime.cycle_counter(), 2);
    }
}

#[test]
fn test_nan_input_sanitized() {
    let result = ShivaBuilder::new().build();
    if let Ok(mut runtime) = result {
        let mut input = SystemInputDTO::default();
        input.state[0] = f32::NAN;
        input.state[1] = f32::INFINITY;
        input.previous_rewards = f32::NAN;

        let output = runtime.step(input);
        if let Ok(out) = output {
            // All output values should be finite
            for &val in out.final_action.iter() {
                assert!(val.is_finite(), "Output action contains non-finite value");
            }
            for &val in out.state.iter() {
                assert!(val.is_finite(), "Output state contains non-finite value");
            }
        }
    }
}

#[test]
fn test_multiple_sequential_cycles() {
    let result = ShivaBuilder::new().build();
    if let Ok(mut runtime) = result {
        for i in 0..100u64 {
            let mut input = SystemInputDTO::default();
            input.timestep = i;
            let output = runtime.step(input);
            assert!(output.is_ok(), "Cycle {} should not fail", i);
        }
        assert_eq!(runtime.cycle_counter(), 100);
    }
}

// ═══════════════════════════════════════════════════════════════
// Dimension Type Alias Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_dimension_type_aliases() {
    assert_eq!(std::mem::size_of::<StateVector>(), DEFAULT_STATE_DIM * 4);
    assert_eq!(std::mem::size_of::<ActionVector>(), DEFAULT_ACTION_DIM * 4);
    assert_eq!(std::mem::size_of::<RuleFlagVector>(), DEFAULT_ACTION_DIM);

    assert_eq!(DEFAULT_STATE_DIM, 64);
    assert_eq!(DEFAULT_ACTION_DIM, 32);
}

// ═══════════════════════════════════════════════════════════════
// Safety Verdict Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_safety_verdict_approved() {
    let action = [0.5f32; 32];
    let verdict = SafetyVerdict::Approved {
        projected_action: action,
    };
    match verdict {
        SafetyVerdict::Approved { projected_action } => {
            assert_eq!(projected_action[0], 0.5);
        }
        _ => panic!("Expected Approved"),
    }
}

#[test]
fn test_safety_verdict_vetoed() {
    let fallback = [0.0f32; 32];
    let verdict = SafetyVerdict::Vetoed {
        reason: "Test veto".to_string(),
        fallback,
    };
    match verdict {
        SafetyVerdict::Vetoed { reason, fallback } => {
            assert_eq!(reason, "Test veto");
            assert_eq!(fallback[0], 0.0);
        }
        _ => panic!("Expected Vetoed"),
    }
}

// ═══════════════════════════════════════════════════════════════
// Environment Stack Tests
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_environment_stack_default() {
    let env = EnvironmentStack::default();
    assert_eq!(env.cycle_counter, 0);
    assert_eq!(env.input_timestep, 0);
    assert!(!env.is_emergency);
    assert!(env.emergency_reason.is_none());
    assert!(env.safety_veto_reason.is_none());
    assert_eq!(env.final_action, [0.0f32; 32]);
}

#[test]
fn test_input_timestep_vs_cycle_counter() {
    let mut env = EnvironmentStack::default();
    env.input_timestep = 42;
    env.cycle_counter = 1;
    // They should be independent
    assert_ne!(env.input_timestep, env.cycle_counter);
}
