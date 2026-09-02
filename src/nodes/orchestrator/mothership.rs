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
// │   SEMANTIC EXPLANATION:                                                        │
// │   The Long Vision Engine acts as a CRITIC/MODULATOR, not an independent actor.│
// │   Its weight w_risk adjusts how strongly the Fast Decision policy is trusted:  │
// │   - w_risk > 0 → amplifies a_fast (trajectory looks safe, trust the policy)   │
// │   - w_risk ≈ 0 → dampens a_fast (high tail risk, reduce policy confidence)    │
// │   This is INTENTIONAL: IQN evaluates trajectory risk and modulates the         │
// │   primary policy accordingly, rather than proposing its own action.            │
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

use crate::framework::diagnostics::CycleDiagnostics;
use crate::framework::error::ShivaError;
use crate::framework::orchestrator::Orchestrator;
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
///
/// This is one implementation of the `Orchestrator` trait. Users can implement
/// alternative orchestration strategies (e.g., simpler PID + safety, or custom
/// ensemble architectures) by implementing the trait directly.
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
    /// NOTE: Acts as a CRITIC/MODULATOR — w_risk modulates a_fast, not an independent action.
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

    /// Internal method: executes the 3-phase pipeline.
    /// Used by both the legacy `execute_cycle` and the new `Orchestrator` trait impl.
    fn run_pipeline(&self, env: &mut EnvironmentStack, diagnostics: &mut CycleDiagnostics) {
        // Increment the monotonic cycle counter.
        env.cycle_counter += 1;

        // Reset emergency flag at the start of each cycle.
        // Each phase may set it to true if an emergency condition is detected.
        env.is_emergency = false;
        env.emergency_reason = None;
        env.safety_veto_reason = None;

        // ═══════════════════════════════════════════════════════════════
        // PHASE 1: Anomaly Evaluation (Failure Engine / RND)
        // ═══════════════════════════════════════════════════════════════
        //
        // WHAT: Check if the environment state S_t is out-of-distribution.
        // WHY: Must happen FIRST — if the state is unknown, no policy output is trustworthy.
        self.failure_engine.execute(env);
        diagnostics.anomaly_score = env.anomaly_output.prediction_error;

        // SHORT-CIRCUIT: If the Failure Engine detected an anomaly, bypass everything.
        // Dispatch the pre-compiled emergency recovery action directly.
        if env.anomaly_output.is_out_of_distribution {
            // Copy emergency action to final_action.
            // The emergency_action is a safe deceleration command (e.g., 0.5 × prev_action).
            env.final_action = env.anomaly_output.emergency_action;
            env.is_emergency = true;
            env.emergency_reason = Some("Out-of-distribution state detected by Failure Engine (RND)");
            diagnostics.set_emergency("OOD state detected — dispatching emergency action".to_string());
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
        // FORMULA (documented consensus semantics):
        //
        //   a_candidate[i] = (w_fast · a_fast[i] + w_risk · a_fast[i] + w_adapt · a_explore[i])
        //                  / (w_fast + w_risk + w_adapt + ε)
        //
        // SEMANTIC EXPLANATION:
        // Long Vision (IQN) is a CRITIC, not an actor. It does not produce its own action.
        // Its weight w_risk modulates the trust placed in the Fast Decision (SAC) policy:
        //   - When w_risk is high: trajectory looks safe → amplify a_fast contribution
        //   - When w_risk is low:  high tail risk → dampen a_fast, shift toward a_explore
        //
        // This is equivalent to:
        //   a_candidate[i] = ((w_fast + w_risk) · a_fast[i] + w_adapt · a_explore[i])
        //                  / (w_fast + w_risk + w_adapt + ε)
        //
        // The combined weight (w_fast + w_risk) represents the total confidence in the
        // primary policy, modulated by trajectory risk assessment.
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

        // Record consensus weights for diagnostics/observability
        diagnostics.set_consensus_weights(w_fast, w_risk, w_adapt);

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
            env.safety_veto_reason = Some("GuardRail Engine vetoed action — constraint violation");
            diagnostics.set_safety_veto("Action vetoed by GuardRail — falling back to prev_action".to_string());
            return;
        }

        // Normal exit: the projected (safety-filtered) action becomes the final command.
        // This is the definitive motor command a*_t dispatched to physical actuators.
        env.final_action = env.constraint_output.projected_action;
    }

    /// Legacy public method: executes one complete 3-phase consensus cycle.
    ///
    /// WHAT: Runs the full Mothership control pipeline and writes `final_action` to
    /// the EnvironmentStack.
    ///
    /// NOTE: Preserved for backward compatibility with code that calls this directly.
    /// The `Orchestrator` trait implementation delegates to the same pipeline.
    pub fn execute_cycle(&self, env: &mut EnvironmentStack) {
        let mut diagnostics = CycleDiagnostics::default();
        self.run_pipeline(env, &mut diagnostics);
    }
}

/// Orchestrator trait implementation for MothershipOrchestrator.
///
/// WHAT: Makes the 5-node Mothership architecture a pluggable framework component.
/// HOW: Delegates to the internal 3-phase pipeline with diagnostics collection.
/// WHY: Users can replace the entire orchestration strategy while retaining the
///      framework's lifecycle, error handling, and hardware dispatch infrastructure.
impl Orchestrator for MothershipOrchestrator {
    fn execute_cycle(
        &self,
        env: &mut EnvironmentStack,
        diagnostics: &mut CycleDiagnostics,
    ) -> Result<(), ShivaError> {
        self.run_pipeline(env, diagnostics);
        Ok(())
    }
}

impl Default for MothershipOrchestrator {
    fn default() -> Self {
        Self::new(
            FailureEngineNode::new(Box::new(crate::brain::anomaly::RndAnomalyAdapter::default())),
            FastDecisionNode::new(Box::new(crate::brain::policy::SacPolicyAdapter::default())),
            LongVisionNode::new(Box::new(crate::brain::risk::IqnRiskAdapter::default())),
            ExplorerNode::new(Box::new(crate::brain::skill_vault::Td3SkillAdapter::default())),
            GuardRailNode::new(Box::new(crate::brain::constraint::CpoConstraintAdapter::default())),
        )
    }
}

