// Shiva 2.0 — Actuator Signal Output Adapter
//
// WHAT THIS FILE DOES:
// Wraps the existing ActuatorSignal hardware dispatch buffer behind the
// OutputAdapter trait, decoupling the core runtime from direct hardware access.
//
// HOW IT DOES IT:
// Implements OutputAdapter::dispatch() by delegating to ActuatorSignal::write()
// and ActuatorSignal::finalSendToSystem().
//
// WHY WE DO THIS:
// ManInTheMiddle currently writes directly to ActuatorSignal and calls
// finalSendToSystem(). This adapter allows the framework to use different
// hardware transports (CAN, EtherCAT, ROS, simulation, serial) by swapping
// the OutputAdapter implementation.

use crate::config::ShivaConfig;
use crate::environment::actuatorSignal::ActuatorSignal;
use crate::framework::adapter::{DispatchResult, OutputAdapter};
use crate::framework::dimensions::ActionVector;
use crate::framework::error::{ShivaError, TransportError};

/// Default OutputAdapter wrapping the ActuatorSignal hardware dispatch buffer.
///
/// WHAT: Bridges the framework's OutputAdapter trait to the existing ActuatorSignal.
/// HOW: Delegates dispatch() to ActuatorSignal::write() + finalSendToSystem().
/// WHY: Preserves backward compatibility while enabling pluggable hardware transports.
pub struct ActuatorSignalAdapter {
    signal: ActuatorSignal,
}

impl ActuatorSignalAdapter {
    /// Creates a new adapter with default ActuatorSignal configuration.
    pub fn new() -> Self {
        Self {
            signal: ActuatorSignal::__init__(),
        }
    }

    /// Creates a new adapter from explicit ShivaConfig parameters.
    pub fn from_config(config: &ShivaConfig) -> Self {
        Self {
            signal: ActuatorSignal::from_config(config),
        }
    }
}

impl Default for ActuatorSignalAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputAdapter for ActuatorSignalAdapter {
    /// Dispatches the final safe action to the ActuatorSignal hardware buffer.
    ///
    /// WHAT: Stages the action in the hardware buffer and transmits it.
    /// HOW: Calls write() to stage, then finalSendToSystem() to dispatch.
    /// WHY: Propagates dispatch errors instead of silently discarding them.
    fn dispatch(&mut self, action: &ActionVector) -> Result<DispatchResult, ShivaError> {
        self.signal.write(action);
        match self.signal.finalSendToSystem() {
            Ok(dispatched) => Ok(DispatchResult {
                dispatched_action: dispatched,
            }),
            Err(e) => Err(ShivaError::Transport(TransportError::DispatchFailed(
                e.to_string(),
            ))),
        }
    }
}
