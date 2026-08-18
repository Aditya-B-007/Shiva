// Shiva 2.0 — Protocol Sub-Module
//
// WHAT THIS MODULE DOES:
// Defines the communication protocol and man-in-the-middle orchestration layer
// connecting external system environments with Shiva 2.0 control nodes.

pub mod systemSide;
pub mod shivaSide;
pub mod middleMan;

pub use systemSide::SystemInputDTO;
pub use shivaSide::ShivaOutputDTO;
pub use middleMan::ManInTheMiddle;
