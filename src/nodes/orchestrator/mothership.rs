// Mothership Orchestrator — 3-Phase Consensus Pipeline for Shiva 2.0
//
// WHAT THIS FILE DOES:
// Implements the MothershipOrchestrator, the central execution coordinator for the
// 5-Node Mothership Ensemble. It runs a deterministic 3-phase consensus pipeline
// on every control cycle, producing a single safe final action for hardware dispatch.
//
// HOW IT DOES IT:
// Holds all 5 engine nodes and executes them in strict sequential order with
// short-circuit logic for emergency conditions. Exposes a single public method
// `execute_cycle()` that processes one complete control loop iteration.
//
// THE 3-PHASE CONSENSUS PIPELINE:
//
// ┌───────────────────────────────────────────────────────────────────────────────┐
// │ PHASE 1: Anomaly Evaluation (Failure Engine / RND)                          │
// │                                                                               │
// │   The Failure Engine measures prediction error E(S_t).                        │
// │   If E(S_t) > threshold_OOD:                                                 │
// │     → BYPASS all other engines                                                │
// │     → Dispatch a_emergency directly to hardware                               │
// │     → RETURN (short-circuit)                                                  │
// ├───────────────────────────────────────────────────────────────────────────────┤
// │ PHASE 2: Unconstrained Candidate Consensus                                   │
// │   (Fast Decision + Long Vision + Explorer Engines)                            │
// │                                                                               │
// │   1. Fast Decision Engine (SAC) → a_fast, w_fast                              │
// │   2. Long Vision Engine (IQN)   → w_risk                                      │
// │   3. Explorer Engine (TD3 + z)  → a_explore, w_adapt                          │
// │                                                                               │
// │   MATHEMATICAL REDUCTION FORMULA:                                             │
// │                                                                               │
// │     a_candidate[i] = (w_fast · a_fast[i] + w_risk · a_fast[i]                 │
// │                      + w_adapt · a_explore[i])                                │
// │                      / (w_fast + w_risk + w_adapt + ε)                        │
// │                                                                               │
// │   WHY NORMALIZED REDUCTION:                                                   │
// │   The denominator (w_fast + w_risk + w_adapt + ε) ensures the net weight      │
// │   coefficient is strictly normalized to ~1.0, preventing accidental torque     │
// │   loss or signal attenuation (magnitude collapse).                             │
// ├───────────────────────────────────────────────────────────────────────────────┤
// │ PHASE 3: Safety Projection Filter (GuardRail Engine / CPO)                    │
// │                                                                               │
// │   The GuardRail Engine acts as an immutable post-pass safety shield:           │
// │   1. Slew-Rate Limiting: |a_t[i] - a_{t-1}[i]| ≤ Δ_max                       │
// │   2. Rule-Mask Filtering: if m_t[i] == 1, zero out channel i                  │
// │   3. Convex Boundary Projection: clamp to [-1.0, 1.0]                         │
// │                                                                               │
// │   If vetoed → fall back to prev_action                                        │
// │   Otherwise → final_action = constraint_output.projected_action               │
// └───────────────────────────────────────────────────────────────────────────────┘
//
// WHY WE DO THIS:
// The 3-phase pipeline ensures that:
// - Unknown/catastrophic states are caught FIRST (Phase 1)
// - Multiple AI engines reach weighted consensus without magnitude collapse (Phase 2)
// - No unchecked command ever reaches physical hardware (Phase 3)

use crate::nodes::core::shared_state::EnvironmentStack;
use crate::nodes::failure_engine::FailureEngineNode;
use crate::nodes::fast_decision::FastDecisionNode;
use crate::nodes::long_vision::LongVisionNode;
use crate::nodes::explorer::ExplorerNode;
use crate::nodes::guardrail::GuardRailNode;

/// MothershipOrchestrator — central coordinator of the 5-Node Ensemble.
///
/// WHAT: Runs the deterministic 3-phase consensus pipeline on every control cycle.
/// HOW: Holds all 5 engine nodes and executes them in strict order with short-circuits.
/// WHY: Guarantees that every control cycle produces exactly one safe, normalized
/// action command for hardware dispatch within sub-millisecond timing constraints.
pub struct MothershipOrchestrator {
    /// Phase 1: Failure Engine (RND anomaly detection).
    /// WHAT: First line of defense — detects out-of-distribution states.
    /// WHY: Must execute before any policy computation to protect against unknowns.
    failure_engine: FailureEngineNode,

    /// Phase 2a: Fast Decision Engine (SAC policy).
    /// WHAT: Generates baseline motor proposal a_fast with confidence w_fast.
    fast_decision: FastDecisionNode,

    /// Phase 2b: Long Vision Engine (IQN risk evaluation).
    /// WHAT: Evaluates trajectory tail-risk and produces risk-adjusted weight w_risk.
    long_vision: LongVisionNode,

    /// Phase 2c: Explorer Engine (TD3 + z skill adaptation).
    /// WHAT: Produces drift-compensated skill action a_explore with weight w_adapt.
    explorer: ExplorerNode,

    /// Phase 3: GuardRail Engine (CPO safety projection).
    /// WHAT: Final safety filter — slew-rate, rule-mask, and boundary projection.
    guardrail: GuardRailNode,
}

impl MothershipOrchestrator {
    /// Creates a new MothershipOrchestrator with all 5 engine nodes.
    ///
    /// WHAT: Constructor assembling the complete 5-node ensemble.
    /// HOW: Accepts pre-constructed engine nodes (already wired to brain facades).
    /// WHY: The orchestrator does not create engines — it orchestrates them.
    /// This follows the Dependency Inversion Principle: construction is external,
    /// execution is internal.
    pub fn new(
        failure_engine: FailureEngineNode,
        fast_decision: FastDecisionNode,
        long_vision: LongVisionNode,
        explorer: ExplorerNode,
        guardrail: GuardRailNode,
    ) -> Self {
        Self {
            failure_engine,
            fast_decision,
            long_vision,
            explorer,
            guardrail,
        }
    }

    /// Single public method: executes one complete 3-phase consensus cycle.
    ///
    /// WHAT: Runs the full Mothership control pipeline and writes `final_action` to
    /// the EnvironmentStack.
    ///
    /// HOW: Executes the 3-phase pipeline described at the top of this file:
    ///   Phase 1 → anomaly check (short-circuit if OOD)
    ///   Phase 2 → consensus of Fast Decision + Long Vision + Explorer
    ///   Phase 3 → safety projection via GuardRail (short-circuit if vetoed)
    ///
    /// MATHEMATICAL MERGE (Phase 2):
    ///   a_candidate[i] = \frac{w_{fast} \cdot a_{fast}[i] + w_{risk} \cdot a_{fast}[i] + w_{adapt} \cdot a_{explore}[i]}{w_{fast} + w_{risk} + w_{adapt} + \epsilon}
    ///
    /// WHY: Produces exactly one deterministic, safe, normalized action per cycle.
    pub fn execute_cycle(&self, env: &mut EnvironmentStack) {
        // Increment the monotonic cycle counter.
        env.cycle_counter += 1;

        // Reset emergency flag at the start of each cycle.
        // Each phase may set it to true if an emergency condition is detected.
        env.is_emergency = false;

        // ═══════════════════════════════════════════════════════════════
        // PHASE 1: Anomaly Evaluation (Failure Engine / RND)
        // ═══════════════════════════════════════════════════════════════
        //
        // WHAT: Check if the environment state S_t is out-of-distribution.
        // WHY: Must happen FIRST — if the state is unknown, no policy output is trustworthy.
        self.failure_engine.execute(env);

        // SHORT-CIRCUIT: If the Failure Engine detected an anomaly, bypass everything.
        // Dispatch the pre-compiled emergency recovery action directly.
        if env.anomaly_output.is_out_of_distribution {
            // Copy emergency action to final_action.
            // The emergency_action is a safe deceleration command (e.g., 0.5 × prev_action).
            env.final_action = env.anomaly_output.emergency_action;
            env.is_emergency = true;
            return;
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 2: Unconstrained Candidate Consensus
        //   (Fast Decision + Long Vision + Explorer Engines)
        // ═══════════════════════════════════════════════════════════════
        //
        // WHAT: Compute normalized weighted reduction across 3 active proposal engines.
        // WHY: Produces an unconstrained candidate command a_candidate that reflects
        //      the best consensus of policy, risk, and adaptation signals.

        // Phase 2a: Fast Decision Engine (SAC) → a_fast, w_fast
        self.fast_decision.execute(env);

        // Phase 2b: Long Vision Engine (IQN) → w_risk
        self.long_vision.execute(env);

        // Phase 2c: Explorer Engine (TD3 + z) → a_explore, w_adapt
        self.explorer.execute(env);

        // MERGE: Normalized weighted reduction to produce a_candidate.
        //
        // FORMULA:
        //   a_candidate[i] = (w_fast · a_fast[i] + w_risk · a_fast[i] + w_adapt · a_explore[i])
        //                  / (w_fast + w_risk + w_adapt + ε)
        //
        // WHY NORMALIZED:
        //   The denominator (w_fast + w_risk + w_adapt + ε) strictly normalizes the
        //   weight coefficient sum to ~1.0. This prevents:
        //   - Magnitude collapse: if all weights are small, action amplitude would shrink
        //   - Signal attenuation: torque loss from under-weighted proposals
        //   - Unbounded growth: if weights sum > 1.0, action could exceed physical limits
        let w_fast = env.policy_output.confidence_weight;
        let w_risk = env.risk_output.risk_adjusted_weight;
        let w_adapt = env.adaptation_output.adaptation_weight;

        // Small epsilon to prevent division by zero in degenerate edge cases.
        let epsilon = 1e-8_f32;
        let weight_sum = w_fast + w_risk + w_adapt + epsilon;

        for i in 0..32 {
            let a_fast = env.policy_output.proposed_action[i];
            let a_explore = env.adaptation_output.adapted_action[i];

            // Weighted consensus: Fast Decision and Long Vision both vote for a_fast,
            // Explorer votes for a_explore. The risk weight w_risk amplifies or dampens
            // the policy proposal based on trajectory safety assessment.
            let numerator = w_fast * a_fast + w_risk * a_fast + w_adapt * a_explore;
            env.candidate_action[i] = numerator / weight_sum;
        }

        // ═══════════════════════════════════════════════════════════════
        // PHASE 3: Safety Projection Filter (GuardRail Engine / CPO)
        // ═══════════════════════════════════════════════════════════════
        //
        // WHAT: Project a_candidate onto the safe convex set.
        // WHY: The candidate from Phase 2 is unconstrained — it has not been validated
        //      against slew-rate limits, rule masks, or physical actuator boundaries.
        //      The GuardRail Engine is the final immutable safety barrier.
        self.guardrail.execute(env);

        // SHORT-CIRCUIT: If the GuardRail engine vetoed the action, fall back to prev_action.
        // A veto means the candidate was so far outside safe bounds that even projection
        // could not produce a legal command (e.g., all channels under hardware interlock).
        if env.constraint_output.is_vetoed {
            env.final_action = env.prev_action;
            env.is_emergency = true;
            return;
        }

        // Normal exit: the projected (safety-filtered) action becomes the final command.
        // This is the definitive motor command a*_t dispatched to physical actuators.
        env.final_action = env.constraint_output.projected_action;
    }
}
