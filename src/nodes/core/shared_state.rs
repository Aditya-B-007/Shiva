// Shiva 2.0 — Shared Memory Environment Stack
//
// WHAT THIS FILE DOES:
// Defines the `EnvironmentStack` — the central C-contiguous, cache-line aligned
// shared memory structure that all 5 Mothership Engine Nodes read from and write to
// during each execution cycle.
//
// HOW IT DOES IT:
// Uses `#[repr(C, align(64))]` layout with fixed-size stack arrays (`[f32; 64]`,
// `[f32; 32]`, `[u8; 32]`) to guarantee zero dynamic heap allocations on the hot path.
// Each engine writes to its dedicated output slot; the orchestrator reads all slots
// and computes `final_action`.
//
// WHY WE DO THIS:
// The 5-Node Mothership Ensemble requires sub-millisecond cycle times. Dynamic heap
// allocation (`Vec`, `String`, `Box`) would introduce unpredictable latency spikes
// from the allocator. By keeping everything on the stack in a single contiguous block,
// we achieve deterministic timing and maximize CPU cache utilization (64-byte cache
// line alignment).
//
// MEMORY LAYOUT:
// ┌─────────────────────────────────────────────────────────┐
// │                    INPUT REGION                         │
// │  current_state: [f32; 64]     ← sensor / env input     │
// │  prev_action:   [f32; 32]     ← last executed action    │
// │  rule_flags:    [u8; 32]      ← hardware safety masks   │
// │  active_skill_id: [u8; 32]   ← skill embedding ID      │
// │  state_history: [f32; 64]    ← recent state window      │
// │  action_history: [f32; 32]   ← recent action window     │
// ├─────────────────────────────────────────────────────────┤
// │               PHASE 1 OUTPUT (Failure Engine)           │
// │  anomaly_output: AnomalyAssessment                      │
// ├─────────────────────────────────────────────────────────┤
// │               PHASE 2 OUTPUTS (Consensus Engines)       │
// │  policy_output:     PolicyProposal     ← Fast Decision  │
// │  risk_output:       RiskAssessment     ← Long Vision    │
// │  adaptation_output: AdaptedProposal    ← Explorer       │
// │  candidate_action:  [f32; 32]          ← merged Phase 2 │
// ├─────────────────────────────────────────────────────────┤
// │               PHASE 3 OUTPUT (GuardRail Engine)         │
// │  constraint_output: ConstraintResult                    │
// ├─────────────────────────────────────────────────────────┤
// │               FINAL OUTPUT                              │
// │  final_action:  [f32; 32]    ← dispatched to actuators  │
// │  cycle_counter: u64          ← monotonic counter        │
// │  is_emergency:  bool         ← emergency flag           │
// └─────────────────────────────────────────────────────────┘

use crate::brain::core::dto::{
    AdaptedProposal, AnomalyAssessment, ConstraintResult, PolicyProposal, RiskAssessment,
};

/// EnvironmentStack — C-contiguous shared memory for the 5-Node Mothership Ensemble.
///
/// WHAT: Central data structure connecting all engine nodes in a single execution cycle.
///
/// HOW: Each engine reads its required input fields and writes to its dedicated output slot.
/// The orchestrator reads all output slots and computes `final_action` via weighted consensus.
///
/// WHY: Zero-allocation shared memory ensures deterministic sub-millisecond cycle times.
/// The `#[repr(C, align(64))]` layout guarantees cache-line alignment for optimal
/// CPU utilization and enables direct FFI interop with C/C++ co-processors if needed.
#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct EnvironmentStack {
    // ═══════════════════════════════════════════════════════
    // INPUT REGION — Written by external environment / sensors before each cycle
    // ═══════════════════════════════════════════════════════

    /// Raw sensor / environment state vector for the current timestep t.
    /// WHAT: The full observation vector S_t from the environment.
    /// WHY: Primary input consumed by Fast Decision, Explorer, and Failure engines.
    pub current_state: [f32; 64],

    /// Previously executed motor action from cycle t-1.
    /// WHAT: The action a*_{t-1} that was actually dispatched to hardware last cycle.
    /// WHY: Needed for rate-of-change limiting (GuardRail) and policy delta confidence.
    pub prev_action: [f32; 32],

    /// Hardware safety interlock bitmask.
    /// WHAT: Per-channel binary flags m_t indicating which actuator channels are locked.
    /// HOW: If rule_flags[i] == 1, channel i is under hardware safety interlock.
    /// WHY: Enables the GuardRail Engine to zero-out torque on over-tempered joints.
    pub rule_flags: [u8; 32],

    /// Active latent skill embedding identifier.
    /// WHAT: 32-byte skill ID selecting which sub-behavior z the Explorer Engine activates.
    /// HOW: Converted to a 16-float latent vector z ∈ [-1, 1]^16 by the Explorer.
    /// WHY: Enables hierarchical skill-conditioned control (TD3 + z).
    pub active_skill_id: [u8; 32],

    /// Flattened recent state history window for trajectory risk evaluation.
    /// WHAT: Sliding window of recent states [S_{t-k}, ..., S_{t-1}] flattened into 64 floats.
    /// WHY: The Long Vision Engine (IQN) evaluates multi-step tail risk over this history.
    pub state_history: [f32; 64],

    /// Flattened recent action history window for trajectory risk evaluation.
    /// WHAT: Sliding window of recent actions [a_{t-k}, ..., a_{t-1}] flattened into 32 floats.
    /// WHY: Paired with state_history for the Long Vision Engine's CVaR computation.
    pub action_history: [f32; 32],

    // ═══════════════════════════════════════════════════════
    // PHASE 1 OUTPUT — Failure Engine (RND Anomaly Detection)
    // ═══════════════════════════════════════════════════════

    /// Output from the Failure Engine's anomaly detection.
    /// WHAT: Contains prediction_error, is_out_of_distribution flag, and emergency_action.
    /// HOW: Written by FailureEngineNode during Phase 1.
    /// WHY: If is_out_of_distribution == true, the orchestrator short-circuits to emergency_action.
    pub anomaly_output: AnomalyAssessment,

    // ═══════════════════════════════════════════════════════
    // PHASE 2 OUTPUTS — Consensus Engines (SAC, IQN, TD3+z)
    // ═══════════════════════════════════════════════════════

    /// Output from the Fast Decision Engine (SAC policy).
    /// WHAT: Contains proposed_action a_fast, confidence_weight w_fast, and entropy_score.
    /// HOW: Written by FastDecisionNode during Phase 2.
    pub policy_output: PolicyProposal,

    /// Output from the Long Vision Engine (IQN risk evaluation).
    /// WHAT: Contains cvar_risk_score and risk_adjusted_weight w_risk.
    /// HOW: Written by LongVisionNode during Phase 2.
    pub risk_output: RiskAssessment,

    /// Output from the Explorer Engine (TD3 + z skill adaptation).
    /// WHAT: Contains adapted_action a_explore, adaptation_weight w_adapt, active_skill_id.
    /// HOW: Written by ExplorerNode during Phase 2.
    pub adaptation_output: AdaptedProposal,

    /// Merged Phase 2 consensus candidate action before safety projection.
    /// WHAT: The unconstrained candidate a_candidate computed via normalized weighted reduction.
    /// HOW: a_candidate = (w_fast·a_fast + w_risk·a_fast + w_adapt·a_explore) / (w_fast + w_risk + w_adapt + ε)
    /// WHY: Intermediate result fed into Phase 3 (GuardRail) for safety projection.
    pub candidate_action: [f32; 32],

    // ═══════════════════════════════════════════════════════
    // PHASE 3 OUTPUT — GuardRail Engine (CPO Safety Projection)
    // ═══════════════════════════════════════════════════════

    /// Output from the GuardRail Engine (CPO constraint enforcement).
    /// WHAT: Contains rule_mask, projected_action (safety-clamped), and is_vetoed flag.
    /// HOW: Written by GuardRailNode during Phase 3.
    /// WHY: The projected_action is the final safe-to-execute command.
    pub constraint_output: ConstraintResult,

    // ═══════════════════════════════════════════════════════════════
    // FINAL OUTPUT — Dispatched to actuators / hardware
    // ═══════════════════════════════════════════════════════════════

    /// The final action dispatched to physical actuators after all 3 phases.
    /// WHAT: The definitive motor command a*_t ∈ [-1, 1]^32.
    /// HOW: Set by the MothershipOrchestrator after Phase 3 completes.
    /// WHY: This is the single output that goes to hardware each cycle.
    pub final_action: [f32; 32],

    /// Monotonically increasing execution cycle counter.
    /// WHAT: Tracks how many complete orchestrator cycles have executed.
    /// WHY: Used for diagnostics, logging, and delayed policy update scheduling.
    /// NOTE: This is the FRAMEWORK cycle counter, NOT the external environment timestep.
    pub cycle_counter: u64,

    /// External environment timestep from the input DTO.
    /// WHAT: The timestamp/step-index provided by the external system.
    /// WHY: Preserved separately from cycle_counter so the framework's internal
    ///      cycle count is independent of the external environment clock.
    pub input_timestep: u64,

    /// Emergency flag indicating the system entered a safety fallback state.
    /// WHAT: Set to true if Phase 1 (anomaly) or Phase 3 (constraint veto) triggered.
    /// WHY: Downstream systems can check this flag for emergency telemetry / logging.
    pub is_emergency: bool,

    /// Human-readable reason for the emergency, if triggered.
    /// WHAT: Describes why the emergency was triggered (OOD detection, veto, etc.).
    /// WHY: Enables downstream systems to log and diagnose emergency conditions.
    pub emergency_reason: Option<&'static str>,

    /// Human-readable reason for a safety veto, if triggered.
    /// WHAT: Describes why the safety pipeline vetoed the proposed action.
    /// WHY: Enables downstream systems to log and diagnose safety interventions.
    pub safety_veto_reason: Option<&'static str>,
}

impl Default for EnvironmentStack {
    /// WHAT: Initializes a zeroed-out EnvironmentStack with safe defaults.
    /// HOW: All arrays zeroed, all DTOs set to their own defaults, cycle_counter = 0.
    /// WHY: Provides a clean starting state for the first execution cycle.
    fn default() -> Self {
        Self {
            current_state: [0.0; 64],
            prev_action: [0.0; 32],
            rule_flags: [0; 32],
            active_skill_id: [0; 32],
            state_history: [0.0; 64],
            action_history: [0.0; 32],

            anomaly_output: AnomalyAssessment::default(),

            policy_output: PolicyProposal::default(),
            risk_output: RiskAssessment::default(),
            adaptation_output: AdaptedProposal::default(),
            candidate_action: [0.0; 32],

            constraint_output: ConstraintResult::default(),

            final_action: [0.0; 32],
            cycle_counter: 0,
            input_timestep: 0,
            is_emergency: false,
            emergency_reason: None,
            safety_veto_reason: None,
        }
    }
}
