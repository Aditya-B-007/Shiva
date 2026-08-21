// Shiva 2.0 — Framework Diagnostics & Observability
//
// WHAT THIS FILE DOES:
// Defines the per-cycle diagnostics structure that captures runtime
// observability data from each execution cycle.
//
// HOW IT DOES IT:
// `CycleDiagnostics` is populated during pipeline execution and made
// available to framework users after each cycle completes.
//
// WHY WE DO THIS:
// Framework users need access to runtime diagnostics for monitoring,
// debugging, and telemetry. Observable data includes cycle timing,
// node execution durations, emergency/veto reasons, consensus weights,
// and anomaly scores. Without this, users cannot diagnose runtime
// behavior or implement application-level health monitoring.

/// Per-cycle diagnostics capturing runtime observability data.
///
/// WHAT: Complete diagnostic snapshot from a single execution cycle.
/// HOW: Populated by the orchestrator and runtime during pipeline execution.
/// WHY: Provides the data needed for monitoring, debugging, and telemetry.
#[derive(Debug, Clone)]
pub struct CycleDiagnostics {
    /// The framework's internal cycle counter.
    pub cycle_id: u64,

    /// The external environment timestep from the input.
    pub input_timestep: u64,

    /// Per-node execution timings in nanoseconds: (node_name, duration_ns).
    pub node_timings_ns: Vec<(String, u64)>,

    /// Whether an emergency condition was triggered this cycle.
    pub emergency_triggered: bool,

    /// Human-readable reason for the emergency, if triggered.
    pub emergency_reason: Option<String>,

    /// Whether the safety pipeline vetoed the proposed action.
    pub safety_veto: bool,

    /// Human-readable reason for the safety veto, if triggered.
    pub safety_veto_reason: Option<String>,

    /// Raw anomaly prediction error score from the Failure Engine.
    pub anomaly_score: f32,

    /// Consensus weights from Phase 2 of the pipeline.
    pub consensus_weights: ConsensusWeights,

    /// The lifecycle state at the time of this cycle.
    pub lifecycle_state: String,
}

/// Consensus weight snapshot from Phase 2 of the pipeline.
///
/// WHAT: Records the individual weights used in the normalized weighted reduction.
/// HOW: Extracted from `policy_output`, `risk_output`, and `adaptation_output`.
/// WHY: Allows users to understand how the consensus was formed and which
///      engines had the strongest influence on the final action.
#[derive(Debug, Clone, Copy, Default)]
pub struct ConsensusWeights {
    /// Weight from the Fast Decision Engine (SAC) — w_fast.
    pub w_fast: f32,
    /// Weight from the Long Vision Engine (IQN) — w_risk.
    pub w_risk: f32,
    /// Weight from the Explorer Engine (TD3 + z) — w_adapt.
    pub w_adapt: f32,
}

impl Default for CycleDiagnostics {
    fn default() -> Self {
        Self {
            cycle_id: 0,
            input_timestep: 0,
            node_timings_ns: Vec::new(),
            emergency_triggered: false,
            emergency_reason: None,
            safety_veto: false,
            safety_veto_reason: None,
            anomaly_score: 0.0,
            consensus_weights: ConsensusWeights::default(),
            lifecycle_state: String::from("Unknown"),
        }
    }
}

impl CycleDiagnostics {
    /// Creates a new diagnostics snapshot for the given cycle.
    pub fn new(cycle_id: u64, input_timestep: u64) -> Self {
        Self {
            cycle_id,
            input_timestep,
            ..Default::default()
        }
    }

    /// Records a node's execution timing.
    pub fn record_node_timing(&mut self, node_name: String, duration_ns: u64) {
        self.node_timings_ns.push((node_name, duration_ns));
    }

    /// Marks this cycle as having triggered an emergency.
    pub fn set_emergency(&mut self, reason: String) {
        self.emergency_triggered = true;
        self.emergency_reason = Some(reason);
    }

    /// Marks this cycle as having been vetoed by the safety pipeline.
    pub fn set_safety_veto(&mut self, reason: String) {
        self.safety_veto = true;
        self.safety_veto_reason = Some(reason);
    }

    /// Records the consensus weights from Phase 2.
    pub fn set_consensus_weights(&mut self, w_fast: f32, w_risk: f32, w_adapt: f32) {
        self.consensus_weights = ConsensusWeights {
            w_fast,
            w_risk,
            w_adapt,
        };
    }
}
