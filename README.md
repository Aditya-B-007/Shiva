<div align="center">

# shiva.ai
### An Open-Source Cognitive Operating System & Sub-Millisecond Autonomous Runtime

*"Building AI that doesn't just predict—it perceives, remembers, regulates, plans, projects safety, and acts in sub-millisecond real time."*![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Language](https://img.shields.io/badge/language-Rust%201.75+-orange.svg)
![Runtime](https://img.shields.io/badge/runtime-Sub--Millisecond-brightgreen.svg)
![Architecture](https://img.shields.io/badge/architecture-5--Node%20Mothership-purple.svg)
![Pipeline](https://img.shields.io/badge/pipeline-3--Phase%20Consensus-blueviolet.svg)
![Status](https://img.shields.io/badge/status-active%20development-green.svg)

</div>
---

![shiva ai](https://github.com/user-attachments/assets/24f134a5-aae8-4044-879a-ae0c431e08b4)

# Why Shiva?

Artificial Intelligence has made remarkable progress in language understanding, reasoning, vision, and code generation. Modern Large Language Models (LLMs) can solve complex problems, write software, and interact with humans in natural ways. Despite these advances, nearly all current AI systems remain fundamentally **reactive**.

They generate responses conditioned on inputs but do not continuously perceive their environment, regulate internal cognitive processes, enforce physical safety constraints, maintain persistent identity, or autonomously decide when and how to act.

Shiva explores a fundamentally different direction: a **Cognitive Operating System (Cognitive OS)** paired with **Shiva 2.0**—a zero-LLM, sub-millisecond autonomous real-time continuous control runtime written in Rust.

---

# The 3-Layer Decoupled Architecture (Shiva 2.0)

Shiva 2.0 strictly enforces the **Dependency Inversion Principle** (SOLID) across three decoupled software layers, ensuring zero dynamic heap allocations on real-time execution hot paths (`#[repr(C, align(64))]`).

```mermaid
graph TB
    subgraph Nodes["Layer 3: src/nodes/ — Domain Layer & Ensemble"]
        FE["FailureEngineNode (Phase 1)"]
        FD["FastDecisionNode (Phase 2)"]
        LV["LongVisionNode (Phase 2)"]
        EX["ExplorerNode (Phase 2)"]
        GR["GuardRailNode (Phase 3)"]
        MO["MothershipOrchestrator"]
    end

    subgraph Brain["Layer 2: src/brain/ — Middleware & Traits"]
        AD["AnomalyDetector Trait"]
        PE["PolicyEvaluator Trait"]
        RE["RiskEvaluator Trait"]
        AE["AdaptationEvaluator Trait"]
        CE["ConstraintEvaluator Trait"]
    end

    subgraph Algorithms["Layer 1: src/algorithms/ — RL Math"]
        RND["Random Network Distillation (RND)"]
        SAC["Soft Actor-Critic (SAC)"]
        IQN["Implicit Quantile Networks (IQN)"]
        TD3["TD3 + Latent Skill Vector (z)"]
        CPO["Constrained Policy Optimization (CPO)"]
    end

    FE -->|depends on| AD
    FD -->|depends on| PE
    LV -->|depends on| RE
    EX -->|depends on| AE
    GR -->|depends on| CE

    AD -.->|facade adapter| RND
    PE -.->|facade adapter| SAC
    RE -.->|facade adapter| IQN
    AE -.->|facade adapter| TD3
    CE -.->|facade adapter| CPO
```

---

# The 5-Node Mothership Ensemble & 3-Phase Consensus Pipeline

At the core of Shiva 2.0's real-time control loop is the **5-Node Mothership Ensemble**, coordinated by the **3-Phase Consensus Pipeline** inside `MothershipOrchestrator`:

```mermaid
graph TD
    ENV["Environment Sensors"] --> ES["EnvironmentStack<br/>#[repr(C, align(64))]"]

    subgraph P1["Phase 1: Anomaly Gate"]
        FE["Failure Engine (RND)"]
    end

    subgraph P2["Phase 2: Unconstrained Candidate Consensus"]
        FD["Fast Decision Engine (SAC)<br/>a_fast, w_fast"]
        LV["Long Vision Engine (IQN)<br/>w_risk"]
        EX["Explorer Engine (TD3+z)<br/>a_explore, w_adapt"]
        MERGE["Normalized Weighted Reduction<br/>a_candidate = (w_fast·a_fast + w_risk·a_fast + w_adapt·a_explore) / (w_fast + w_risk + w_adapt + ε)"]
    end

    subgraph P3["Phase 3: Immutable Safety Shield"]
        GR["GuardRail Engine (CPO)<br/>• Slew-Rate Limiting |Δa| ≤ Δ_max<br/>• Rule-Mask Interlocks m_t<br/>• Convex Set Boundary Projection"]
    end

    ES --> FE
    FE -->|OOD Trigger E(S_t) > Threshold| EMG["Emergency Recovery Action<br/>a_emergency (Short-Circuit)"]
    FE -->|Nominal State| FD
    FE -->|Nominal State| LV
    FE -->|Nominal State| EX

    FD --> MERGE
    LV --> MERGE
    EX --> MERGE

    MERGE --> GR
    GR -->|Vetoed| PREV["Hold Previous Action<br/>a*_{t-1} (Safety Fallback)"]
    GR -->|Safe Projected Command| HW["Final Hardware Dispatch<br/>a*_t ∈ [-1, 1]³²"]
```

---
��─────────────────────────────────┐
 │ PHASE 2: Unconstrained Candidate Consensus                      │
 │ 1. Fast Decision Engine (SAC): Baseline motor proposal a_fast   │
 │ 2. Long Vision Engine (IQN): Multi-step CVaR_α risk weight w_risk│
 │ 3. Explorer Engine (TD3 + z): Drift-compensated action a_explore│
 │                                                                 │
 │ Mathematical Reduction Formula (Prevents Magnitude Collapse):  │
 │              w_fast · a_fast + w_risk · a_fast + w_adapt · a_explore │
 │ a_candidate = ───────────────┬───────────────────────────────── │
 │                 w_fast + w_risk + w_adapt + ε                   │
 └────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │ PHASE 3: Safety Projection Filter (GuardRail Engine / CPO)      │
 │ • Slew-Rate Limiting: |a_t - a_{t-1}| ≤ Δ_max                   │
 │ • Rule-Mask Filtering: Hardware safety interlocks m_t           │
 │ • Convex Boundary Projection onto Safe Legal Set                │
 └────────────────────────────────┬────────────────────────────────┘
                                  │
                                  ▼
                     Final Action (Hardware Dispatch)
```

---

# Cognitive Operating System (Python / High-Level Cognition)

While Shiva 2.0 handles sub-millisecond continuous control in Rust, the high-level cognitive system coordinates perception, memory, homeostasis, and reasoning:

```text
                    Environment
                          │
                          ▼
                 Multi-Modal Perception
          (Text • Vision • Audio • Sensors)
                          │
                          ▼
               Semantic Understanding (BERT)
                          │
                          ▼
               Orchestration & Event System
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
      Cognitive Swarm             Environment Interface
   (Multiple Cognitive Nodes)             (Shiva 2.0 Rust Runtime)
          │                                       │
          ▼                                       ▼
  Emotion • Memory • Scratchpad           Sub-Millisecond
  Identity • Planning • Goals             Continuous Control
          │
          ▼
 Shared Working Memory & Coordination
          │
          ▼
      Reasoning Engine (SmolLM2 / Frankenmerged Models)
          │
          ▼
 Tool Selection & Action Execution
```

---

# Core Concepts & Algorithms

### 1. Soft Actor-Critic (SAC) — Fast Decision Engine
- **Objective**: Maximum Entropy RL maximizing expected return and entropy:
  $$J(\pi) = \sum_{t=0}^T \mathbb{E}_{(s_t, a_t)} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$
- **Role**: Provides the baseline continuous motor policy proposal $a_{\text{fast}}$ with policy confidence $w_{\text{fast}}$.

### 2. Implicit Quantile Networks (IQN) — Long Vision Engine
- **Objective**: Distributional RL modeling quantile return function $Z_\tau(s, a)$ with cosine embeddings $\psi_j(\tau) = \text{ReLU}\left(\sum_{i=0}^{K-1} \cos(i \pi \tau) w_{ij} + b_j\right)$.
- **Role**: Computes Conditional Value-at-Risk ($\text{CVaR}_\alpha$) over lower-tail returns to derive risk weight $w_{\text{risk}}$.

### 3. TD3 + Latent Skill Embedding ($z$) — Explorer Engine
- **Objective**: Twin Delayed DDPG conditioned on latent skill vector $z \in \mathbb{R}^{16}$ with target policy smoothing and delayed updates.
- **Role**: Detects physical parameter drift (friction, mass, wind) and produces drift-compensated action $a_{\text{explore}}$.

### 4. Constrained Policy Optimization (CPO) — GuardRail Engine
- **Objective**: Enforces safety constraints $J^C(\theta) \le d_k$ under KL bounds using convex projection.
- **Role**: Immutable post-pass safety filter enforcing slew-rate limits $|a_t - a_{t-1}| \le \Delta_{\text{max}}$, rule masks $m_t$, and boundary clamping.

### 5. Random Network Distillation (RND) — Failure Engine
- **Objective**: Feature prediction error between target network $f^*(s)$ and predictor network $\hat{f}_\theta(s)$:
  $$E(s) = \frac{1}{k} \|\hat{f}_\theta(s) - f^*(s)\|_2^2$$
- **Role**: Out-of-distribution (OOD) novelty detector triggering emergency safety fallbacks.

---

# Codebase Structure So Far

```text
Shiva/
├── Cargo.toml                  # Rust Crate Manifest
├── README.md                   # System Architecture & Documentation
├── src/
│   ├── lib.rs                  # Crate Root (algorithms, brain, nodes)
│   ├── algorithms/             # Layer 1: Pure RL Mathematical Implementations
│   │   ├── mod.rs
│   │   ├── softActorCriticNetwork.rs
│   │   ├── cpo.rs
│   │   ├── implicitQuantileNetworks.rs
│   │   ├── td3.rs
│   │   └── rnd.rs
│   ├── brain/                  # Layer 2: Decoupling Middleware & Facades
│   │   ├── mod.rs
│   │   ├── core/
│   │   │   ├── mod.rs
│   │   │   ├── dto.rs          # SIMD-aligned C DTOs (align(64))
│   │   │   └── traits.rs       # 5 Core Brain Trait Contracts
│   │   ├── policy/
│   │   │   ├── mod.rs
│   │   │   └── facade.rs       # SacPolicyAdapter
│   │   ├── constraint/
│   │   │   ├── mod.rs
│   │   │   └── facade.rs       # CpoConstraintAdapter
│   │   ├── risk/
│   │   │   ├── mod.rs
│   │   │   └── facade.rs       # IqnRiskAdapter
│   │   ├── skill_vault/
│   │   │   ├── mod.rs
│   │   │   └── facade.rs       # Td3SkillAdapter
│   │   └── anomaly/
│   │       ├── mod.rs
│   │       └── facade.rs       # RndAnomalyAdapter
│   └── nodes/                  # Layer 3: Domain Execution & Orchestrator
│       ├── mod.rs
│       ├── core/
│       │   ├── mod.rs
│       │   └── shared_state.rs # EnvironmentStack Shared Memory
│       ├── failure_engine/
│       │   ├── mod.rs
│       │   └── node.rs         # FailureEngineNode (Phase 1)
│       ├── fast_decision/
│       │   ├── mod.rs
│       │   └── node.rs         # FastDecisionNode (Phase 2)
│       ├── long_vision/
│       │   ├── mod.rs
│       │   └── node.rs         # LongVisionNode (Phase 2)
│       ├── explorer/
│       │   ├── mod.rs
│       │   └── node.rs         # ExplorerNode (Phase 2)
│       ├── guardrail/
│       │   ├── mod.rs
│       │   └── node.rs         # GuardRailNode (Phase 3)
│       └── orchestrator/
│           ├── mod.rs
│           └── mothership.rs   # MothershipOrchestrator (3-Phase Pipeline)
```

---

# Technology Stack

### Real-Time Runtime (Shiva 2.0)
- **Language**: Rust (Edition 2021)
- **Memory Management**: Zero dynamic heap allocations on hot paths (`#[repr(C, align(64))]` stack arrays)
- **Architecture**: 3-Layer Decoupled Architecture enforcing Dependency Inversion
- **Algorithms**: SAC, CPO, IQN, TD3 + $z$, RND

---


# Mission

> **Shiva is not another AI assistant built around a language model. It is a Cognitive Operating System paired with a sub-millisecond real-time continuous control engine that transforms existing hardware into autonomous cognitive agents capable of continuously perceiving, reasoning, remembering, planning, projecting safety, and acting.**

---

## Contributing
Shiva is an open-source research project. We welcome contributions in Deep Learning, Reinforcement Learning, Systems Programming (Rust), Cognitive Science, and Robotics.

## License
Released under the Apache 2.0 License.

---
**Intelligence is more than prediction. It is perception, regulation, memory, adaptation, safety projection, and action.**
