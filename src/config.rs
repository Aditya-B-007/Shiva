// Shiva 2.0 — Framework Configuration & Builder Module
//
// WHAT THIS FILE DOES:
// Defines `ShivaConfig` and `ShivaBuilder` providing a 3-tier configuration hierarchy:
// 1. Programmatic Builder Setting (Highest priority)
// 2. Environment Variable (`SHIVA_MATRIX_ROWS`, `SHIVA_ACTUATOR_MIN`, `SHIVA_ACTUATOR_MAX`)
// 3. Framework Defaults (20 rows, [-1.0, 1.0] signal limits)

use std::env;
use crate::protocol::middleMan::ManInTheMiddle;

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
}

impl Default for ShivaConfig {
    fn default() -> Self {
        Self::auto_detect()
    }
}

/// Builder pattern for configuring and constructing the Shiva `ManInTheMiddle` protocol orchestrator.
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

    /// Builds and initializes the configured `ManInTheMiddle` framework instance.
    pub fn build(self) -> ManInTheMiddle {
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
}
