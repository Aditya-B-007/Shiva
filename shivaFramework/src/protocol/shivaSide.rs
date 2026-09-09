// Shiva 2.0 — Shiva Side Protocol DTO
//
// WHAT THIS FILE DOES:
// Defines `ShivaOutputDTO`, the data transfer packet returned from Shiva 2.0 back to the system environment.
//
// HOW IT DOES IT:
// Encapsulates 4 output parameters resulting from the 3-phase consensus pipeline:
// 1. state: Goal-conditioned / processed state vector [f32; 64]
// 2. reward: Computed / normalized step reward (f32)
// 3. mask: Active safety interlock bitmask [u8; 32]
// 4. final_action: Safe motor command vector dispatched to physical actuators [f32; 32]

/// Data Transfer Object representing processed output sent back to the external system.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShivaOutputDTO {
    /// Goal-conditioned processed state vector S_t
    pub state: [f32; 64],
    /// Evaluated step reward scalar R_t
    pub reward: f32,
    /// Active safety rule bitmask m_t
    pub mask: [u8; 32],
    /// Final safe motor action command vector a*_t dispatched to actuators
    pub final_action: [f32; 32],
}

impl ShivaOutputDTO {
    /// Creates a new ShivaOutputDTO packet.
    pub fn new(
        state: [f32; 64],
        reward: f32,
        mask: [u8; 32],
        final_action: [f32; 32],
    ) -> Self {
        Self {
            state,
            reward,
            mask,
            final_action,
        }
    }
}

impl Default for ShivaOutputDTO {
    fn default() -> Self {
        Self {
            state: [0.0; 64],
            reward: 0.0,
            mask: [0; 32],
            final_action: [0.0; 32],
        }
    }
}
