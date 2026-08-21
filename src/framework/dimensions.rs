// Shiva 2.0 — Dimension Type Aliases
//
// WHAT THIS FILE DOES:
// Defines type aliases for the state, action, and rule-flag vectors used
// throughout the Shiva framework. This replaces hard-coded `[f32; 64]`,
// `[f32; 32]`, and `[u8; 32]` with named types.
//
// HOW IT DOES IT:
// Provides compile-time constants for default dimensions and type aliases
// that reference them. All framework code imports these aliases instead of
// using raw array types directly.
//
// WHY WE DO THIS:
// A framework should support different observation/action spaces. By
// centralizing dimension definitions, we create a single point of change
// for future migration to const generics or configurable dimensions.
// This is the incremental approach: non-breaking today, extensible tomorrow.

/// Default state observation vector dimension.
///
/// WHAT: The number of f32 elements in a state observation vector S_t.
/// WHY: Matches the 64-float observation space used in the current architecture.
pub const DEFAULT_STATE_DIM: usize = 64;

/// Default action output vector dimension.
///
/// WHAT: The number of f32 elements in a motor action command vector a_t.
/// WHY: Matches the 32-float action space used in the current architecture.
pub const DEFAULT_ACTION_DIM: usize = 32;

/// Default latent skill embedding dimension.
///
/// WHAT: The number of f32 elements in the latent skill vector z.
/// WHY: Matches the 16-float latent space used by the Explorer Engine (TD3 + z).
pub const DEFAULT_SKILL_DIM: usize = 16;

/// State observation vector type alias.
///
/// WHAT: Fixed-size array representing the full environment observation S_t.
/// WHY: Provides a named type instead of bare `[f32; 64]` throughout the codebase.
pub type StateVector = [f32; DEFAULT_STATE_DIM];

/// Action command vector type alias.
///
/// WHAT: Fixed-size array representing a motor action command a_t.
/// WHY: Provides a named type instead of bare `[f32; 32]` throughout the codebase.
pub type ActionVector = [f32; DEFAULT_ACTION_DIM];

/// Hardware rule-flag bitmask type alias.
///
/// WHAT: Per-channel binary flags indicating hardware safety interlocks.
/// HOW: If `rule_flags[i] == 1`, actuator channel i is under interlock.
/// WHY: Provides a named type instead of bare `[u8; 32]` throughout the codebase.
pub type RuleFlagVector = [u8; DEFAULT_ACTION_DIM];

/// Skill embedding identifier type alias.
///
/// WHAT: 32-byte identifier selecting which sub-behavior z the Explorer activates.
/// WHY: Provides a named type instead of bare `[u8; 32]` throughout the codebase.
pub type SkillIdVector = [u8; DEFAULT_ACTION_DIM];
