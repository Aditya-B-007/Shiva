<div align="center">

# shiva.ai
### An Open-Source Cognitive Operating System & Sub-Millisecond Autonomous Runtime

*"Building AI that doesn't just predict—it perceives, remembers, regulates, plans, projects safety, and acts in sub-millisecond real time."*

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
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
    subgraph Nodes["Layer 3: src/nodes/ - Domain Layer & Ensemble"]
        FE["FailureEngineNode (Phase 1)"]
        FD["FastDecisionNode (Phase 2)"]
        LV["LongVisionNode (Phase 2)"]
        EX["ExplorerNode (Phase 2)"]
        GR["GuardRailNode (Phase 3)"]
        MO["MothershipOrchestrator"]
    end

    subgraph Brain["Layer 2: src/brain/ - Middleware & Traits"]
        AD["AnomalyDetector Trait"]
        PE["PolicyEvaluator Trait"]
        RE["RiskEvaluator Trait"]
        AE["AdaptationEvaluator Trait"]
        CE["ConstraintEvaluator Trait"]
    end

    subgraph Algorithms["Layer 1: src/algorithms/ - RL Math"]
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