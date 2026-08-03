<div align="center">

# shiva.ai
### An Open-Source Cognitive Architecture for General intelligence.

*"Building AI that doesn't just predict—it perceives, remembers, regulates, plans, and acts."*

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![Status](https://img.shields.io/badge/status-active%20development-orange.svg)
![Research](https://img.shields.io/badge/research-cognitive%20AI-purple.svg)

</div>



![shiva ai](https://github.com/user-attachments/assets/24f134a5-aae8-4044-879a-ae0c431e08b4)

# Why Shiva?

Shiva is a command-line cognitive human-like agent which acts as an integrated cognitive model rather than just a simple wrapper around an AI model. In a nutshell, it is general intelligence designed to run directly inside developer workspaces.

Rather than building another conversational chatbot, Shiva aims to build a Cognitive General Intelligence model capable of continuously planning, reasoning, executing code, remembering, and adapting directly inside developer workspaces.

---

# The Problem

Most workspace AI tools today are purely reactive—the agents rely on probabilistic ways of coding and generating language without maintaining long-term state, dynamic regulation, or cognitive feedback loops.

Intelligence should not begin with a prompt and end with a response. Real-world autonomous General Intelligence systems must continuously think, reason, remember previous executions, self-critique, and execute sandbox operations without manual prompt-chaining.

---

# Our Approach

Shiva models workspace automation as a **distributed Cognitive Operating System** rather than a monolithic neural network. It decomposes cognition into specialized subsystems responsible for distinct aspects of intelligence, coordinating them through a shared runtime.

The language model is **not the brain**. It is a **reasoning engine** used by the cognitive system whenever probabilistic inference, planning, or code generation is required.

---

# Detailed Model Architecture & How It Works

Shiva operates as a multi-stage cognitive loop composed of specialized modules across perception, memory, regulation, reasoning swarm, and reinforcement learning.

```text
                 [ Dev Workspace / CLI ]
                   (src/input/cli.py)
                            │
                            ▼
                  Workspace Context Hook
               (src/input/hook/workspace.py)
                            │
                            ▼
             Perception & Context Formatting
              (src/input/hook/perception.py)
                            │
                            ▼
                Mothership Arbitration Loop
               (src/swarm/mothership.py)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
     Cognitive Swarm              Internal Regulation
   (src/swarm/cells.py)     (Appraisal & Homeostasis Engine)
            │                               │
            ▼                               ▼
    Chain of Thought               Cognitive Stability
  (Thought • Critique)            (Pendulum Physics Control)
            │
            ▼
    Sandbox Execution ──────────► SQLite MemoryGraph
  (src/brain/transformer/     (src/brain/memory/MemoryEngine.py)
        Decoder.py)                         │
                                            ▼
                                Monte Carlo Credit Assignment
                              (src/brain/memory/algorithms/
                                     CreditAssigner.py)
```

---

### Subsystem Deep-Dive & Source Map

#### 1. Perception & Workspace Integration
- **Entry Point**: [`cli.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/input/cli.py) initializes the cognitive engine, sets up fallback orchestrators, and launches the user loop.
- **Workspace Hook**: [`workspace.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/input/hook/workspace.py) scans user project directories, indexing files, git states, and directory structures.
- **Perception Formatter**: [`perception.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/input/hook/perception.py) bundles raw inputs and workspace observations into structured DTOs defined in [`transferDTO.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/transferDTO.py).

#### 2. Prefrontal Arbitration: Mothership & Stability Regulator
- **Mothership Core**: [`mothership.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/mothership.py) acts as the central executive coordinator. It manages cell dispatch, pheromone trail decay (confidence tracking), neuro-symbolic value estimation, and memory persistence.
- **Cognitive Stability Regulator**: Integrated within [`mothership.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/mothership.py#L29-L56), this utilizes an inverted pendulum physical model (`theta` angular deviation) driven by uncertainty, stress, and conflict. The Mothership applies stabilizing prefrontal control effort to prevent cognitive instability and hallucination drift.

#### 3. Cognitive Swarm & Cortical Columns
- **Cortical Columns**: [`cells.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/cells.py) implements specialized cortical columns:
  - **`AnalyticalColumn`**: Focuses on logical decomposition and step-by-step code structure.
  - **`CreativeColumn`**: Generates alternative solutions, edge-case hypotheses, and out-of-the-box approaches.
  - **`RiskColumn`**: Assesses potential regressions, destructive file ops, or missing safety constraints.
  - **`VerificationColumn`**: Evaluates syntax correctness, test validity, and plan alignment.
- **Reinforcement Learning Swarm**: [`SwarmSAC.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/SwarmSAC.py) provides a Soft Actor-Critic (SAC) continuous policy network to optimize column selection weights and action allocations.

#### 4. Emotional Regulation & Homeostasis
- **Appraisal Engine**: [`AppraisalEngine.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/emotionalHandlerAndStore/AppraisalEngine.py) uses FT-Transformer models to evaluate task context across cognitive dimensions (goal relevance, urgency, novelty).
- **Emotion Dynamics**: [`EmotionDynamicsEngine.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/emotionalHandlerAndStore/EmotionDynamicsEngine.py) tracks valence, arousal, and cognitive load dynamics over time.
- **Homeostasis Engine**: [`Homeostasis.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/emotionalHandlerAndStore/Homeostasis.py) maintains systemic stability (energy, stress, equilibrium thresholds).

#### 5. Memory Engine & SQLite Persistence
- **Memory Engine**: [`MemoryEngine.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/MemoryEngine.py) orchestrates episodic memory encoding, retrieval, spreading activation, and trajectory logging.
- **Topological Graph**: [`MemoryGraph.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/graph/MemoryGraph.py), [`MemoryNode.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/graph/MemoryNode.py), and [`MemoryEdge.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/graph/MemoryEdge.py) form a vector-less graph with activation spreading and association decay.
- **SQLite Persistence**: [`SQLiteMemoryRepository.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/infrastructure/sqlite/SQLiteMemoryRepository.py) and [`SQLiteManager.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/infrastructure/sqlite/SQLiteManager.py) handle WAL-mode transaction-safe graph state persistence.

#### 6. Autonomous Idle Processing (Sleep & Dream Cycles)
- **Sleep & Consolidation**: [`SleepCycle.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/SleepCycle.py) and [`Consolidator.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/Consolidator.py) merge short-term memory nodes into long-term structures during user idle time.
- **Dream Generator**: [`DreamGenerator.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/DreamGenerator.py) synthesizes hypothetical trajectories to refine memory association strengths before prompt execution resumes.
- **Forgetting & Identity**: [`ForgettingModel.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/ForgettingModel.py) prunes low-activation nodes, while [`IdentityUpdater.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/IdentityUpdater.py) updates high-level agent identity parameters.

#### 7. Reinforcement & Credit Assignment
- **Monte Carlo Credit Assigner**: [`CreditAssigner.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/CreditAssigner.py) evaluates completed trajectory episodes and reinforces nodes and edges mathematically:
  $$\Delta V(S) = \alpha (G - V(S))$$
  This directly updates node `strength` and edge `association_strength` in SQLite based on task outcome rewards.

#### 8. Sandbox Execution & Output
- **Thought Parsing**: [`thought_parser.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/transformer/thought_parser.py) extracts `THOUGHT`, `CRITIQUE`, `CONFIDENCE`, and `DECISION` blocks from reasoning chains.
- **Code Decoder & Sandbox**: [`Decoder.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/transformer/Decoder.py) compiles code blocks and safely executes them in a local workspace sandbox environment.
- **Voice Synthesis Output**: [`voice.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/output/voice.py) provides optional speech generation for response outputs.

---

# Technology Stack

### Cognition & AI Core
- **Memory**: Custom vector-less SQLite `MemoryGraph` with topological activation spreading ([`MemoryEngine.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/MemoryEngine.py)).
- **Regulation**: Prefrontal `Mothership` coordinator with physical pendulum stability regulation ([`mothership.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/mothership.py)).
- **Appraisal**: FT-Transformer-backed emotional dynamics appraisal engine ([`AppraisalEngine.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/emotionalHandlerAndStore/AppraisalEngine.py)).
- **Reinforcement Learning**: Soft Actor-Critic (SAC) continuous policy optimization ([`SwarmSAC.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/SwarmSAC.py)).
- **DTO Layer**: Slotted PyTorch & NumPy data structures ([`transferDTO.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/transferDTO.py)).

### Runtime & Language
- **Language**: Python 3.11+
- **Database**: SQLite3 with WAL mode ([`SQLiteManager.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/infrastructure/sqlite/SQLiteManager.py))

---

# Mission

> **Shiva is a Cognitive Operating System that transforms existing developer workspaces into autonomous cognitive runtimes, continuously planning, executing, remembering, and adapting.**

<img width="1051" height="881" alt="Screenshot 2026-07-11 at 11 25 46 PM" src="https://github.com/user-attachments/assets/e3c32e2a-2f01-403a-a807-db6fe3b56d0a" />

---

## Contributing
Shiva is an open-source research project. We welcome contributions in Deep Learning, Reinforcement Learning, Cognitive Science, and Software Automation. 

## License
Released under the Apache 2.0 License.

---
**Intelligence is more than prediction. It is perception, regulation, memory, adaptation, and action.**

