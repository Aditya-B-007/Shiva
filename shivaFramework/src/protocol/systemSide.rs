// Shiva 2.0 — System Side Protocol DTO
//
// WHAT THIS FILE DOES:
// Defines `SystemInputDTO`, the data transfer packet received from the external system environment.
//
// HOW IT DOES IT:
// Encapsulates 6 key input parameters required for continuous control, safety checks, and risk analysis:
// 1. state: Raw environment observation state [f32; 64]
// 2. setpoint: Goal / target control trajectory [f32; 32]
// 3. state_stack: Recent state history window [f32; 64]
// 4. action_stack: Recent action history window [f32; 32]
// 5. hard_boundaries: Hardware safety interlock bitmask [u8; 32]
// 6. previous_rewards: Reward feedback scalar from previous cycle (f32)
// 7. timestep: Monotonic cycle clock (u64)

/// Data Transfer Object representing incoming state & control inputs from the external system.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SystemInputDTO {
    /// Raw state observation vector S_t from environment sensors
    pub state: [f32; 64],
    /// Target control setpoint / desired goal state
    pub setpoint: [f32; 32],
    /// Sliding window history of recent environment states
    pub state_stack: [f32; 64],
    /// Sliding window history of recent executed actions
    pub action_stack: [f32; 32],
    /// Hardware safety interlock bitmask (1 = locked/fault, 0 = legal)
    pub hard_boundaries: [u8; 32],
    /// Reward feedback scalar R_{t-1} from the environment
    pub previous_rewards: f32,
    /// Monotonic system clock timestamp / step cycle index
    pub timestep: u64,
}

impl SystemInputDTO {
    /// Creates a new SystemInputDTO with custom parameters.
    pub fn new(
        state: [f32; 64],
        setpoint: [f32; 32],
        state_stack: [f32; 64],
        action_stack: [f32; 32],
        hard_boundaries: [u8; 32],
        previous_rewards: f32,
        timestep: u64,
    ) -> Self {
        Self {
            state,
            setpoint,
            state_stack,
            action_stack,
            hard_boundaries,
            previous_rewards,
            timestep,
        }
    }
}

impl Default for SystemInputDTO {
    fn default() -> Self {
        Self {
            state: [0.0; 64],
            setpoint: [0.0; 32],
            state_stack: [0.0; 64],
            action_stack: [0.0; 32],
            hard_boundaries: [0; 32],
            previous_rewards: 0.0,
            timestep: 0,
        }
    }
}
