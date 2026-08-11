// GuardRail Engine Node — Phase 3 of the 3-Phase Consensus Pipeline
//
// WHAT THIS FILE DOES:
// Implements the GuardRail Engine (Phase 3) of the Mothership Ensemble.
// Acts as an immutable post-pass safety shield that enforces physical actuator limits
// on the unconstrained candidate action a_candidate from Phase 2.
//
// HOW IT DOES IT:
// Holds a `Box<dyn ConstraintEvaluator>` trait object from `src/brain/core/traits.rs`.
// Exposes a single public method `execute()` that reads candidate_action, prev_action,
// and rule_flags from the EnvironmentStack, calls `evaluate_constraints()`, and writes
// the result to `env.constraint_output`.
//
// SAFETY OPERATIONS:
// 1. Slew-Rate Limiting (\Delta a_t): Reads the previously executed action a*_{t-1}
//    and limits instantaneous command jumps to prevent actuator chatter or mechanical wear.
//    |a_t[i] - a_{t-1}[i]| \leq \Delta_{max}
//
// 2. Rule-Mask Filtering (m_t): Applies binary hardware safety interlocks.
//    If m_t[i] == 1, channel i is zeroed (e.g., over-tempered motor joint).
//
// 3. Convex Boundary Projection: If a_candidate lies outside safe physical parameters,
//    the GuardRail Engine mathematically projects it onto the nearest legal convex boundary:
//    a^*_t = \text{GuardRailEngine.project\_to\_safe\_set}(a_{candidate}, a^*_{t-1}, m_t)
//
// WHY WE DO THIS:
// The candidate command from Phase 2 is unconstrained — it reflects optimal AI
// exploration and drift compensation but has NOT been validated against physical
// actuator limits. The GuardRail Engine is the final safety barrier before hardware
// dispatch, ensuring no command can damage physical actuators or violate safety rules.

use crate::brain::core::traits::ConstraintEvaluator;
use crate::nodes::core::shared_state::EnvironmentStack;

/// GuardRailNode — Phase 3 immutable safety projection filter.
///
/// WHAT: Enforces slew-rate limits, rule-mask filtering, and convex boundary projection.
/// HOW: Delegates to a `Box<dyn ConstraintEvaluator>` wrapping the CPO algorithm.
/// WHY: Final safety barrier preventing unsafe commands from reaching physical actuators.
pub struct GuardRailNode {
    /// Trait object wrapping the CPO constraint evaluation implementation.
    /// WHAT: The brain-layer adapter implementing constraint enforcement.
    /// WHY: Dependency inversion — this node never imports from `crate::algorithms`.
    evaluator: Box<dyn ConstraintEvaluator>,
}

impl GuardRailNode {
    /// Creates a new GuardRailNode with the given constraint evaluator implementation.
    ///
    /// WHAT: Constructor accepting any `ConstraintEvaluator` trait implementor.
    /// HOW: Wraps the evaluator in a `Box<dyn ConstraintEvaluator>` for dynamic dispatch.
    /// WHY: Allows swapping safety projection strategies without changing node code.
    pub fn new(evaluator: Box<dyn ConstraintEvaluator>) -> Self {
        Self { evaluator }
    }

    /// Single public method: executes Phase 3 safety projection.
    ///
    /// WHAT: Projects the unconstrained a_candidate onto the safe convex set.
    ///
    /// HOW:
    /// 1. Reads `env.candidate_action` (unconstrained Phase 2 output),
    ///    `env.prev_action` (a*_{t-1}), and `env.rule_flags` (safety bitmask m_t).
    /// 2. Calls `evaluator.evaluate_constraints(candidate_action, prev_action, rule_flags)`
    ///    via the brain trait.
    /// 3. Writes the resulting `ConstraintResult` to `env.constraint_output`.
    /// 4. If `is_vetoed == true`, sets `env.is_emergency = true`.
    ///
    /// The evaluator internally performs:
    ///   - Slew-rate clamping: |a_t[i] - a_{t-1}[i]| ≤ Δ_max
    ///   - Rule-mask zeroing: if m_t[i] == 1, a_t[i] = 0
    ///   - Boundary clamping: a_t[i] ∈ [-1.0, 1.0]
    ///
    /// WHY: The projected_action in constraint_output is the final safe-to-execute
    /// command. If vetoed, the orchestrator falls back to prev_action.
    pub fn execute(&self, env: &mut EnvironmentStack) {
        // Project the unconstrained candidate action through the safety filter.
        // The evaluator applies slew-rate limiting, rule-mask filtering,
        // and convex boundary projection.
        let result = self.evaluator.evaluate_constraints(
            &env.candidate_action,
            &env.prev_action,
            &env.rule_flags,
        );

        // Write result to the shared EnvironmentStack.
        env.constraint_output = result;

        // If the GuardRail engine vetoed the action (e.g., all channels locked),
        // flag the emergency. The orchestrator will fall back to prev_action.
        if result.is_vetoed {
            env.is_emergency = true;
        }
    }
}
