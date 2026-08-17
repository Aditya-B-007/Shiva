// Shiva 2.0 — Actuator Signal Module
//
// WHAT THIS FILE DOES:
// Manages the staging buffer and hardware dispatch pipeline for physical actuator commands.
//
// HOW IT DOES IT:
// - `__init__` / `new`: Configures output channel dimensions, signal limits ([-1.0, 1.0]), and safety clamps.
// - `write`: Stages the final action vector into the hardware dispatch buffer.
// - `finalSendToSystem`: Transmits the staged actuator commands down to the hardware actuator interface.

/// Configuration parameters for ActuatorSignal hardware transmission.
#[derive(Debug, Clone, Copy)]
pub struct ActuatorConfig {
    pub channel_count: usize,
    pub min_signal: f32,
    pub max_signal: f32,
}

impl Default for ActuatorConfig {
    fn default() -> Self {
        Self {
            channel_count: 32,
            min_signal: -1.0,
            max_signal: 1.0,
        }
    }
}

/// ActuatorSignal — Staging and hardware dispatch buffer for motor commands.
#[derive(Debug, Clone)]
pub struct ActuatorSignal {
    pub config: ActuatorConfig,
    /// Staged action signal buffer awaiting hardware dispatch
    staged_signal: [f32; 32],
    /// Flag indicating whether un-dispatched signals are present in the buffer
    is_ready: bool,
}

impl ActuatorSignal {
    /// 2a. __init__ / new: Method for configuring hardware transmission parameters.
    pub fn new(config: ActuatorConfig) -> Self {
        Self {
            config,
            staged_signal: [0.0; 32],
            is_ready: false,
        }
    }

    /// Alias constructor matching requested `__init__` naming convention.
    pub fn __init__() -> Self {
        Self::new(ActuatorConfig::default())
    }

    /// 2b. write: Method to stage an action signal array into the actuator output buffer.
    /// Clamps input signals to configured range [min_signal, max_signal].
    pub fn write(&mut self, action_signal: &[f32; 32]) {
        for i in 0..self.config.channel_count {
            let clamped = action_signal[i].clamp(self.config.min_signal, self.config.max_signal);
            self.staged_signal[i] = clamped;
        }
        self.is_ready = true;
    }

    /// 2c. finalSendToSystem: Method to dispatch staged actuator signals to physical hardware systems.
    /// Returns the dispatched signal array and resets the readiness status.
    pub fn finalSendToSystem(&mut self) -> Result<[f32; 32], &'static str> {
        if !self.is_ready {
            return Err("ActuatorSignal error: No staged action signal to dispatch.");
        }

        let dispatched_signal = self.staged_signal;
        self.is_ready = false; // Reset staging buffer status

        Ok(dispatched_signal)
    }

    /// Idiomatic snake_case alias for `finalSendToSystem`.
    pub fn final_send_to_system(&mut self) -> Result<[f32; 32], &'static str> {
        self.finalSendToSystem()
    }

    /// Returns whether a staged signal is ready for hardware dispatch.
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }
}

impl Default for ActuatorSignal {
    fn default() -> Self {
        Self::__init__()
    }
}
