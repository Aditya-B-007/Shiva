// Shiva 2.0 — Nodes Layer (Domain Execution)
//
// WHAT THIS MODULE DOES:
// Declares and re-exports the 5-Node Mothership Ensemble and the central
// MothershipOrchestrator that coordinates the 3-Phase Consensus Pipeline.
//
// HOW IT IS ORGANIZED:
// ├── core/           → EnvironmentStack (C-contiguous shared memory)
// ├── failure_engine/ → Phase 1: Anomaly gate (RND)
// ├── fast_decision/  → Phase 2: Baseline motor policy (SAC)
// ├── long_vision/    → Phase 2: Trajectory tail-risk (IQN)
// ├── explorer/       → Phase 2: Drift compensation & skills (TD3 + z)
// ├── guardrail/      → Phase 3: Safety projection filter (CPO)
// └── orchestrator/   → MothershipOrchestrator (3-phase pipeline)
//
// WHY WE DO THIS:
// The nodes layer is the domain execution layer. It ONLY depends on
// `src/brain/` trait interfaces — it NEVER imports from `src/algorithms/`.
// This enforces the Dependency Inversion Principle at compile time.

pub mod core;
pub mod failure_engine;
pub mod fast_decision;
pub mod long_vision;
pub mod explorer;
pub mod guardrail;
pub mod orchestrator;

// Re-export shared memory struct
pub use core::EnvironmentStack;

// Re-export engine nodes
pub use failure_engine::FailureEngineNode;
pub use fast_decision::FastDecisionNode;
pub use long_vision::LongVisionNode;
pub use explorer::ExplorerNode;
pub use guardrail::GuardRailNode;

// Re-export orchestrator
pub use orchestrator::MothershipOrchestrator;
