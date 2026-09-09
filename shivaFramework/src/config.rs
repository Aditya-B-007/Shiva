// Shiva 2.0 — Framework Configuration & Builder Module
//
// WHAT THIS FILE DOES:
// Defines `ShivaConfig` and `ShivaBuilder` providing a 3-tier configuration hierarchy:
// 1. Programmatic Builder Setting (Highest priority)
// 2. Environment Variable (`SHIVA_MATRIX_ROWS`, `SHIVA_ACTUATOR_MIN`, `SHIVA_ACTUATOR_MAX`)
// 3. Framework Defaults (20 rows, [-1.0, 1.0] signal limits)
//
// KEY CHANGES (v2):
// - ShivaBuilder::build() now constructs the full MothershipOrchestrator by default
// - Configuration validation prevents invalid parameters (NaN, Inf, min >= max)
// - build() returns Result<ShivaRuntime, ShivaError> instead of ManInTheMiddle
// - Builder supports custom orchestrator, input/output adapter injection
// - The old ManInTheMiddle construction path is preserved via build_legacy()

use std::env;
use crate::framework::error::{ConfigError, ShivaError};
use crate::protocol::middleMan::ManInTheMiddle;
use crate::runtime::ShivaRuntime;

/// Framework Configuration holding parameters for EnvironmentMatrix and ActuatorSignal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShivaConfig {
    /// Maximum capacity of rows in the EnvironmentMatrix sliding window
    pub matrix_rows: usize,
    /// Minimum lower bound for actuator signal output clamping
    pub min_signal: f32,
    /// Maximum upper bound for actuator signal output clamping
    pub max_signal: f32,
}

impl ShivaConfig {
    /// Auto-detects configuration parameters evaluating:
    /// Environment Variables -> Hardcoded Framework Defaults.
    pub fn auto_detect() -> Self {
        let matrix_rows = env::var("SHIVA_MATRIX_ROWS")
            .ok()
            .and_then(|val| val.parse::<usize>().ok())
            .unwrap_or(20);

        let min_signal = env::var("SHIVA_ACTUATOR_MIN")
            .ok()
            .and_then(|val| val.parse::<f32>().ok())
            .unwrap_or(-1.0);

        let max_signal = env::var("SHIVA_ACTUATOR_MAX")
            .ok()
            .and_then(|val| val.parse::<f32>().ok())
            .unwrap_or(1.0);

        Self {
            matrix_rows,
            min_signal,
            max_signal,
        }
    }

    /// Validates configuration parameters and returns an error if invalid.
    ///
    /// WHAT: Checks for NaN, Inf, min >= max, zero rows, etc.
    /// WHY: Invalid configuration should fail during construction, not execution.
    pub fn validate(&self) -> Result<(), ShivaError> {
        // Check for NaN values
        if self.min_signal.is_nan() || self.max_signal.is_nan() {
            return Err(ShivaError::Configuration(ConfigError::NaNOrInfinite {
                field: "actuator_limits",
            }));
        }

        // Check for infinite values
        if self.min_signal.is_infinite() || self.max_signal.is_infinite() {
            return Err(ShivaError::Configuration(ConfigError::NaNOrInfinite {
                field: "actuator_limits",
            }));
        }

        // Check min < max
        if self.min_signal >= self.max_signal {
            return Err(ShivaError::Configuration(ConfigError::InvalidLimits {
                min: self.min_signal,
                max: self.max_signal,
                reason: "min_signal must be strictly less than max_signal",
            }));
        }

        // Check matrix_rows > 0
        if self.matrix_rows == 0 {
            return Err(ShivaError::Configuration(ConfigError::InvalidDimension {
                name: "matrix_rows",
                value: 0,
            }));
        }

        Ok(())
    }
}

impl Default for ShivaConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

/// Builder pattern for configuring and constructing the Shiva runtime.
///
/// WHAT: Constructs a fully-wired ShivaRuntime with validated configuration.
///
/// HOW: Accepts optional overrides for orchestrator, input/output adapters.
///      If not provided, defaults are constructed (MothershipOrchestrator with
///      all 5 engine nodes, DefaultInputAdapter, ActuatorSignalAdapter).
///
/// WHY: The builder ensures that the runtime is always constructed with a
///      mandatory orchestrator and validated configuration. The old path
///      of constructing ManInTheMiddle with `orchestrator: None` is removed.
#[derive(Debug, Clone)]
pub struct ShivaBuilder {
    config: ShivaConfig,
}

impl ShivaBuilder {
    /// Creates a new ShivaBuilder with auto-detected environment configurations.
    pub fn new() -> Self {
        Self {
            config: ShivaConfig::auto_detect(),
        }
    }

    /// Sets the maximum capacity of rows in the EnvironmentMatrix sliding window.
    pub fn with_matrix_rows(mut self, rows: usize) -> Self {
        self.config.matrix_rows = rows;
        self
    }

    /// Sets the actuator signal clamping limits [min_signal, max_signal].
    pub fn with_actuator_limits(mut self, min: f32, max: f32) -> Self {
        self.config.min_signal = min;
        self.config.max_signal = max;
        self
    }

    /// Returns the current configuration (for inspection before building).
    pub fn config(&self) -> &ShivaConfig {
        &self.config
    }

    /// Builds and initializes a fully-wired `ShivaRuntime` with default components.
    ///
    /// WHAT: Constructs the complete runtime with all 5 engine nodes, orchestrator,
    ///       and default adapters.
    ///
    /// HOW:
    /// 1. Validates configuration
    /// 2. Constructs all 5 brain-layer adapters (RND, SAC, IQN, TD3+z, CPO)
    /// 3. Constructs all 5 engine nodes wired to their adapters
    /// 4. Constructs the MothershipOrchestrator with all 5 nodes
    /// 5. Constructs DefaultInputAdapter and ActuatorSignalAdapter
    /// 6. Assembles the ShivaRuntime
    ///
    /// WHY: The default build path now produces a FULLY FUNCTIONAL runtime
    ///      with the complete V2 execution pipeline. No more `orchestrator: None`.
    pub fn build(self) -> Result<ShivaRuntime, ShivaError> {
        // Validate configuration before construction
        self.config.validate()?;

        // Construct default brain-layer adapters
        use crate::brain::anomaly::RndAnomalyAdapter;
        use crate::brain::constraint::CpoConstraintAdapter;
        use crate::brain::policy::SacPolicyAdapter;
        use crate::brain::risk::IqnRiskAdapter;
        use crate::brain::skill_vault::Td3SkillAdapter;

        // Construct engine nodes wired to their adapters
        use crate::nodes::failure_engine::FailureEngineNode;
        use crate::nodes::fast_decision::FastDecisionNode;
        use crate::nodes::long_vision::LongVisionNode;
        use crate::nodes::explorer::ExplorerNode;
        use crate::nodes::guardrail::GuardRailNode;

        let failure = FailureEngineNode::new(Box::new(RndAnomalyAdapter::default()));
        let fast = FastDecisionNode::new(Box::new(SacPolicyAdapter::default()));
        let vision = LongVisionNode::new(Box::new(IqnRiskAdapter::default()));
        let explorer = ExplorerNode::new(Box::new(Td3SkillAdapter::default()));
        let guard = GuardRailNode::new(Box::new(CpoConstraintAdapter::default()));

        // Construct MothershipOrchestrator with all 5 nodes
        use crate::nodes::orchestrator::MothershipOrchestrator;
        let orchestrator = MothershipOrchestrator::new(failure, fast, vision, explorer, guard);

        // Construct default adapters
        use crate::adapters::actuator_adapter::ActuatorSignalAdapter;
        use crate::adapters::default_input::DefaultInputAdapter;

        let output_adapter = ActuatorSignalAdapter::from_config(&self.config);
        let input_adapter = DefaultInputAdapter::new();

        Ok(ShivaRuntime::new(
            self.config,
            Box::new(orchestrator),
            Box::new(output_adapter),
            Box::new(input_adapter),
        ))
    }

    /// Builds the legacy `ManInTheMiddle` for backward compatibility.
    ///
    /// WHAT: Constructs a ManInTheMiddle with the full orchestrator wired in.
    /// WHY: Preserves backward compatibility for existing code that uses
    ///      ManInTheMiddle::counter(). New code should use build() → ShivaRuntime.
    #[deprecated(note = "Use build() for ShivaRuntime instead. ManInTheMiddle is being phased out.")]
    pub fn build_legacy(self) -> ManInTheMiddle {
        ManInTheMiddle::from_config(self.config)
    }
}

impl Default for ShivaBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_overrides_defaults() {
        let builder = ShivaBuilder::new()
            .with_matrix_rows(50)
            .with_actuator_limits(-2.0, 2.0);

        assert_eq!(builder.config.matrix_rows, 50);
        assert_eq!(builder.config.min_signal, -2.0);
        assert_eq!(builder.config.max_signal, 2.0);
    }

    #[test]
    fn test_config_validates_min_gte_max() {
        let config = ShivaConfig {
            matrix_rows: 20,
            min_signal: 1.0,
            max_signal: -1.0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validates_equal_limits() {
        let config = ShivaConfig {
            matrix_rows: 20,
            min_signal: 0.0,
            max_signal: 0.0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validates_nan() {
        let config = ShivaConfig {
            matrix_rows: 20,
            min_signal: f32::NAN,
            max_signal: 1.0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validates_inf() {
        let config = ShivaConfig {
            matrix_rows: 20,
            min_signal: f32::NEG_INFINITY,
            max_signal: 1.0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_config_validates_zero_rows() {
        let config = ShivaConfig {
            matrix_rows: 0,
            min_signal: -1.0,
            max_signal: 1.0,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_valid_config_passes() {
        let config = ShivaConfig {
            matrix_rows: 20,
            min_signal: -1.0,
            max_signal: 1.0,
        };
        assert!(config.validate().is_ok());
    }
}
