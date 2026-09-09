// Shiva 2.0 — Framework Safety Policy Trait
//
// WHAT THIS FILE DOES:
// Defines the `SafetyPolicy` trait — the mandatory safety contract that
// cannot be accidentally omitted from the framework pipeline.
//
// HOW IT DOES IT:
// Provides a `validate_action()` method that evaluates a proposed action
// against the previous action and hardware rule flags, returning either
// an approved (possibly modified) action or a veto with fallback.
//
// WHY WE DO THIS:
// A framework intended for control systems should not allow the central
// safety/orchestration layer to be accidentally omitted. By making safety
// a required trait rather than an optional component, we ensure that every
// action dispatched to hardware has passed through safety validation.

use crate::framework::error::ShivaError;
use crate::framework::dimensions::{ActionVector, RuleFlagVector};

/// The result of a safety policy evaluation.
///
/// WHAT: Represents the safety pipeline's decision on a proposed action.
/// HOW: Either approves (with optional projection) or vetoes the action.
/// WHY: Structured verdict enables the runtime to handle both cases
///      deterministically.
#[derive(Debug, Clone)]
pub enum SafetyVerdict {
    /// The action was approved (possibly after projection/clamping).
    Approved {
        /// The projected safe action to dispatch.
        projected_action: ActionVector,
    },

    /// The action was vetoed — too unsafe to dispatch.
    Vetoed {
        /// Human-readable reason for the veto.
        reason: String,
        /// Fallback action to dispatch instead (typically prev_action).
        fallback: ActionVector,
    },
}

/// Mandatory safety boundary for the framework pipeline.
///
/// WHAT: Trait that enforces physical safety constraints on proposed actions.
///
/// HOW: Evaluates a proposed action against the previous action and hardware
///      rule flags. The implementation may perform:
///      - Slew-rate limiting: |a_t[i] - a_{t-1}[i]| ≤ Δ_max
///      - Rule-mask filtering: if rule_flags[i] == 1, zero channel i
///      - Boundary clamping: a_t[i] ∈ [min_signal, max_signal]
///
/// WHY: The safety pipeline is the last barrier before hardware dispatch.
///      Making it a mandatory trait (not optional) prevents the framework
///      from accidentally operating without safety constraints.
///
/// # Contract
///
/// Implementations MUST guarantee:
/// 1. The `projected_action` in `SafetyVerdict::Approved` satisfies all
///    configured safety constraints
/// 2. The `fallback` in `SafetyVerdict::Vetoed` is a known-safe action
/// 3. The method NEVER panics — all error conditions must be returned
///    as `Err(ShivaError)` or encoded in the verdict
///
/// # Example
///
/// ```rust
/// use shiva::framework::{SafetyPolicy, SafetyVerdict, ShivaError};
/// use shiva::framework::dimensions::{ActionVector, RuleFlagVector};
///
/// struct BasicClampPolicy {
///     min: f32,
///     max: f32,
/// }
///
/// impl SafetyPolicy for BasicClampPolicy {
///     fn validate_action(
///         &self,
///         proposed: &ActionVector,
///         prev: &ActionVector,
///         rule_flags: &RuleFlagVector,
///     ) -> Result<SafetyVerdict, ShivaError> {
///         let mut projected = *proposed;
///         for i in 0..projected.len() {
///             projected[i] = projected[i].clamp(self.min, self.max);
///         }
///         Ok(SafetyVerdict::Approved { projected_action: projected })
///     }
/// }
/// ```
pub trait SafetyPolicy: Send + Sync {
    /// Validates a proposed action against safety constraints.
    ///
    /// WHAT: Evaluates whether the proposed action is safe to dispatch.
    ///
    /// # Arguments
    /// * `proposed` — The candidate action from the consensus pipeline
    /// * `prev` — The previously dispatched action (for slew-rate limiting)
    /// * `rule_flags` — Hardware safety interlock bitmask
    ///
    /// # Returns
    /// * `Ok(SafetyVerdict::Approved { .. })` — Action is safe (possibly modified)
    /// * `Ok(SafetyVerdict::Vetoed { .. })` — Action is unsafe; use fallback
    /// * `Err(ShivaError)` — Safety evaluation itself failed
    fn validate_action(
        &self,
        proposed: &ActionVector,
        prev: &ActionVector,
        rule_flags: &RuleFlagVector,
    ) -> Result<SafetyVerdict, ShivaError>;
}
