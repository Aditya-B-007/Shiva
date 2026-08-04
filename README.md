<div align="center">

# shiva.ai
### An Open-Source Cognitive Architecture for General Intelligence.

*"Building AI that doesn't just predict—it perceives, remembers, regulates, plans, and acts."*

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![Status](https://img.shields.io/badge/status-active%20development-orange.svg)
![Research](https://img.shields.io/badge/research-cognitive%20AI-purple.svg)

</div>

![shiva ai](https://github.com/user-attachments/assets/24f134a5-aae8-4044-879a-ae0c431e08b4)

---

# Complete System Blueprint & Developer Knowledge Base

This document serves as an exhaustive, authoritative blueprint of the **Shiva.ai** repository. Whether you are an AI Coding Agent (e.g. Claude Code, Cursor) or a human contributor, this guide explains every module, file location, data flow, mathematical algorithm, and operational invariant across the codebase.

---

# 1. Executive Summary & Core Philosophy

Shiva is a **distributed Cognitive Operating System** designed to run autonomously inside developer workspaces. Rather than acting as a simple LLM prompt-response wrapper, Shiva models intelligence as specialized cognitive subsystems (Perception, Memory, Regulation, Swarm Reasoning, and Sandbox Execution) running concurrently around a shared state runtime.

### Key Philosophical Invariants:
1. **The LLM is NOT the Brain**: The language model (`HuggingFaceTB/SmolLM2-360M-Instruct`) is a specialized *reasoning engine* invoked by the cognitive system whenever probabilistic inference, planning, or code generation is required.
2. **Deterministic Fallbacks & Zero-Crash Parsing**: No unhandled LLM output formatting error is allowed to crash the system. All outputs pass through a 5-stage resilient parsing pipeline (`ThoughtParser.parse()`).
3. **Hybrid Vector-Graph Memory**: Semantic retrieval combines dense vector similarity (ChromaDB HNSW) with topological graph spreading activation and credit assignment (SQLite).
4. **Physical Regulation of Hallucinations**: Prefrontal arbitration is governed by an inverted physical pendulum model (`CognitiveStabilityRegulator`), translating cognitive load, stress, and uncertainty into physical control effort.

---

# 2. Exhaustive Codebase Directory & File Map

```text
Shiva Project Root (/Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva)
│
├── README.md                           # Master architectural blueprint and knowledge base
├── setup.py                            # Package setup script
├── Shiva.spec                          # PyInstaller desktop packaging configuration
├── logo.ico / logo.jpg                 # Project branding icons
├── .env                                # Environment variables & local runtime configs
│
├── src/                                # Core Python Source Code
│   │
│   ├── transferDTO.py                  # CENTRAL DTO DEFINITIONS (Single Source of Truth)
│   │                                   # - ActionInputDTO (Pydantic: code, command, params)
│   │                                   # - ThoughtStepDTO (Pydantic: reasoning, action, action_input, confidence)
│   │                                   # - ThoughtDTO & ThoughtParseDiagnosticsDTO (dataclasses)
│   │                                   # - ReasoningContextDTO, NodeReasoningResultDTO, ColumnResultDTO
│   │                                   # - MemoryNodeDTO, MemoryEdgeDTO, RetrievalDTO, MemoryGraphDTO
│   │                                   # - PerceptionBundleDTO, PerceptionObservationDTO, PerceptionCaptureRequestDTO
│   │                                   # - FeatureBundle, NumericalFeatureVector, TokenBundle, Tokens, Latent
│   │
│   ├── input/                          # Perception, Workspace Sensing & Entry Points
│   │   ├── cli.py                      # Main CLI loop, terminal interface, fallback cognitive core setup
│   │   └── hook/
│   │       ├── __init__.py             # Input hook exports
│   │       ├── workspace.py            # WorkspaceContext: directory tree scanner, git status, workspace indexing
│   │       └── perception.py           # PerceptionObservationFactory & PerceptionPromptFormatter
│   │
│   ├── brain/                          # Primary Cognitive Brain Subsystems
│   │   │
│   │   ├── transformer/                # Neural Inference, Parsing & Sandbox Execution
│   │   │   ├── Decoder.py              # Causal LM decoder (SmolLM2-360M) & local Python 3 sandbox execution
│   │   │   ├── Encoder.py              # BERT-large feature encoder (AutoTokenizer & AutoModel)
│   │   │   └── thought_parser.py      # Resilient ThoughtParser engine (5-tier fallback cascade)
│   │   │
│   │   ├── node/                       # Single-Node Cognitive Reasoning Loop
│   │   │   ├── scratchPad.py           # Working memory ScratchPad tracking ThoughtStepDTO & SLM prompts
│   │   │   ├── chainOfThought.py       # ChainOfThought iteration counter, convergence & stopping criteria
│   │   │   └── nodeProcessingEngine.py # nodeProcessingEngine: orchestrates Memory, Emotion, ScratchPad, Decoder
│   │   │
│   │   ├── memory/                     # Multi-Tier Memory Engine & Persistence
│   │   │   ├── MemoryEngine.py         # Main memory orchestrator (Hybrid RAG scoring algorithm)
│   │   │   │
│   │   │   ├── graph/                  # Topological Memory Graph Core
│   │   │   │   ├── MemoryNode.py       # MemoryNode (modality, activation, strength, emotional_salience)
│   │   │   │   ├── MemoryEdge.py       # MemoryEdge (association_strength, AssociationType, activation_probability)
│   │   │   │   └── MemoryGraph.py      # Spreading activation, BFS neighbor lookup, activation/deactivation
│   │   │   │
│   │   │   ├── repository/             # Hybrid Memory Persistence Architecture
│   │   │   │   ├── MemoryRepository.py # Abstract base repository interface
│   │   │   │   ├── ChromaMemoryRepository.py # ChromaDB HNSW vector database persistence (shiva_nodes)
│   │   │   │   ├── SQLiteMemoryRepository.py # SQLite relational persistence for graph topology & edges
│   │   │   │   ├── HybridMemoryRepository.py # Hybrid dual-write coordinator
│   │   │   │   └── __init__.py         # Package exports
│   │   │   │
│   │   │   └── algorithms/             # Memory Dynamics & RL Reinforcement
│   │   │       ├── Consolidator.py     # Short-term to long-term memory node consolidation
│   │   │       ├── DreamGenerator.py   # Synthetic trajectory generation during idle sleep
│   │   │       ├── ForgettingModel.py  # Decay & pruning of dormant memory nodes
│   │   │       ├── IdentityUpdater.py  # High-level agent identity refinement
│   │   │       ├── SleepCycle.py       # Background sleep cycle orchestrator
│   │   │       └── CreditAssigner.py   # Monte Carlo temporal reinforcement: ΔV(S) = α (G - V(S))
│   │   │
│   │   ├── emotionalHandlerAndStore/   # Dynamic Emotional & Homeostatic Regulation
│   │   │   ├── emotionInterface.py     # Interfaces (IAppraisal, IFeatureEmbedding, IMemoryEngine)
│   │   │   ├── AppraisalEngine.py      # FT-Transformer contextual appraisal engine & feature extractor
│   │   │   ├── EmotionDynamicsEngine.py# Valence, arousal, and cognitive load dynamics model
│   │   │   ├── Homeostasis.py          # Systemic equilibrium (energy, fatigue, focus, cognitive_load)
│   │   │   └── emotionHandlerAndOrchestrator.py # EmotionalOrchestrator coordinator
│   │   │
│   │   ├── gate/                       # Transactional Gate Architecture & Promotion
│   │   │   ├── Gate.py                 # Main transaction Gate interface
│   │   │   ├── ScratchPadGate.py       # Temporary thought storage gate
│   │   │   ├── MemoryGate.py           # Long-term memory promotion gate
│   │   │   ├── interfaces/             # TransactionManager, MemoryRepository, ScratchRepository
│   │   │   ├── models/                 # WriteRequest, ScratchEntry, Memory
│   │   │   └── services/               # PromotionPolicy & TransactionService
│   │   │
│   │   └── infrastructure/             # Persistence & Queue Infrastructure
│   │       ├── sqlite/
│   │       │   ├── SQLiteManager.py    # Thread-local SQLite connection manager with WAL mode
│   │       │   ├── SQLiteScratchRepository.py # Scratchpad table persistence
│   │       │   └── SQLiteMemoryRepository.py  # Memory table persistence
│   │       └── queue/
│   │           ├── QueueManager.py     # Thread-safe task queue manager
│   │           └── GateWorker.py       # Asynchronous background worker thread
│   │
│   ├── swarm/                          # Distributed Swarm Intelligence & RL Policy
│   │   ├── mothership.py               # Mothership prefrontal coordinator & CognitiveStabilityRegulator
│   │   ├── cells.py                    # Cortical Columns (Analytical, Creative, Risk, Verification)
│   │   └── SwarmSAC.py                 # SwarmSAC: Soft Actor-Critic continuous policy network
│   │
│   ├── output/
│   │   └── voice.py                    # Pyttsx3 / TTS voice synthesis engine
│   │
│   └── packaging/
│       └── desktop/build.py            # Desktop application builder script
│
├── models/                             # Persistent cache for neural models (models/shiva-decoder)
├── chroma_db/                          # Persistent ChromaDB HNSW vector store
└── memory.db                           # SQLite database for relational graph topology & credit logs
```

---

# 3. Subsystem Architecture & Implementation Details

## A. Cognitive Transformer & Decoder ([`src/brain/transformer/Decoder.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/transformer/Decoder.py))
- **Model**: `HuggingFaceTB/SmolLM2-360M-Instruct` (360 Million parameter model, ~720 MB RAM footprint, open & non-gated Apache-2.0 license).
- **Local Caching**: Saved under `models/shiva-decoder` to run fully offline once loaded.
- **Dynamic Projection Layer**: `Decoder.input()` projects arbitrary latent vector dimensions (`vector_input`) to match the decoder model's hidden dimension (`hidden_size = 960`) via a dynamically created `nn.Linear` layer.
- **Local Sandbox Execution**:
  - [`Decoder.execute_code_in_sandbox()`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/transformer/Decoder.py#L140-L177) compiles and executes Python code blocks inside a controlled scope.
  - Exposes standard Python modules (`math`, `json`, `re`, `pathlib`, `shutil`, `glob`, `datetime`, `subprocess`, etc.) and sets `WORKSPACE_DIR` to the active developer project folder.
  - Redirects `stdout` and `stderr` to capture output strings directly into the decision flow.

---

## B. Resilient Multi-Tier Thought Parser ([`src/brain/transformer/thought_parser.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/transformer/thought_parser.py))
- **Public API**: `ThoughtParser.parse(raw_text: str) -> ThoughtStepDTO`
- **Fallback Cascade**:
  1. **Stage 1 (XML `<think>` Stripping)**: Extracts text inside `<think>...</think>` tags (including unclosed `<think>` tags emitted by Qwen/DeepSeek/SmolLM models).
  2. **Stage 2 (Markdown Fence Extraction)**: Extracts JSON inside ` ```json ... ``` ` or ` ``` ... ``` ` blocks.
  3. **Stage 3 (Regex JSON Extraction)**: Scans for raw JSON object candidates (`\{[\s\S]*\}`).
  4. **Stage 4 (JSON Pre-Cleaning & Pydantic Validation)**: Fixes trailing commas (`,\s*}`), replaces Python booleans (`True`/`False`/`None` $\rightarrow$ `true`/`false`/`null`), and validates with `ThoughtStepDTO`.
  5. **Stage 5 (Legacy Tag & Emergency Fallback)**: Parses legacy `THOUGHT:`, `CRITIQUE:`, `CONFIDENCE:`, `DECISION:` tags. If all else fails, constructs `ThoughtStepDTO(reasoning=raw_text, action="scratchpad_note", action_input=ActionInputDTO(raw_output=raw_text), confidence=0.3)`.
- **Zero Crash Guarantee**: Designed to handle any malformed model output without raising uncaught exceptions.

---

## C. Working Memory & ScratchPad ([`src/brain/node/scratchPad.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/node/scratchPad.py))
- **Role**: Maintains working memory state during multi-step reasoning episodes.
- **Key Methods**:
  - `append_thought(thought)`: Accepts strings, `ThoughtDTO`, or `ThoughtStepDTO` objects, automatically routing raw strings through `ThoughtParser.parse()`.
  - `get_prompt_schema_instructions()`: Calls `ThoughtStepDTO.get_json_schema_prompt()` to inject clean Pydantic JSON Schema instructions into SLM prompts.
  - `current_context()`: Assembles `ReasoningContextDTO` for the next decoder iteration.

---

## D. Hybrid Memory Engine ([`src/brain/memory/`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/))
- **Repository Coordinator**: [`HybridMemoryRepository.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/repository/HybridMemoryRepository.py) dual-writes nodes across ChromaDB and SQLite:
  - **ChromaDB** ([`ChromaMemoryRepository.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/repository/ChromaMemoryRepository.py)): Stores composite document vectors (`summary` + `context_signature` + `raw_content`) in an HNSW cosine space (`shiva_nodes` collection).
  - **SQLite** ([`SQLiteMemoryRepository.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/repository/SQLiteMemoryRepository.py)): Stores relational graph edges (`MemoryEdge`), activation states, and credit histories.
- **Hybrid Retrieval Algorithm** ([`MemoryEngine.retrieve()`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/MemoryEngine.py#L82-L115)):
  $$\text{Final Score}(N) = 0.35 \cdot S_{\text{vector}} + 0.25 \cdot S_{\text{keyword}} + 0.15 \cdot \text{Activation} + 0.10 \cdot \text{Strength} + 0.10 \cdot \text{Recency} + 0.05 \cdot \text{Salience}$$
  - $S_{\text{vector}}$: Normalized cosine similarity score from ChromaDB HNSW vector index.
  - $S_{\text{keyword}}$: Token stem intersection ratio and exact match bonus.
  - $\text{Activation} / \text{Strength} / \text{Recency}$: Spreading topological activation along graph edges.

---

## E. Prefrontal Regulation & Cognitive Swarm ([`src/swarm/`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/))
- **Mothership Core**: [`mothership.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/mothership.py) acts as the executive controller.
- **Cognitive Stability Regulator**:
  - Located in `CognitiveStabilityRegulator` ([`mothership.py:29-56`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/mothership.py#L29-L56)), this tracks cognitive instability as an **inverted physical pendulum model** (`theta` angular deviation):
    $$\ddot{\theta} = \frac{g \sin\theta + F_{\text{disturbance}} \cos\theta - F_{\text{control}}}{L}$$
  - Disturbances are driven by uncertainty, stress, and conflict. If $\theta > 0.15 \text{ rad}$, the Mothership dynamically schedules `RiskColumn` to evaluate safety constraints.
- **Cortical Columns** ([`cells.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/cells.py)):
  - `AnalyticalColumn`: Logical decomposition & structured code planning.
  - `CreativeColumn`: Alternative solutions, edge-case hypotheses & novel angles.
  - `RiskColumn`: Safety checks, regression analysis & destructive operation flags.
  - `VerificationColumn`: Syntax validation, plan checking & outcome verification.
- **Reinforcement Learning Swarm Policy** ([`SwarmSAC.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/swarm/SwarmSAC.py)):
  - Implements Soft Actor-Critic (SAC) continuous policy optimization (`SACActor` & `SACCritic`).
  - Maps a 4096-dim state-goal vector to a 2048-dim latent action vector used to condition decoder generation.

---

## F. Idle Processing & Reinforcement Learning Algorithms ([`src/brain/memory/algorithms/`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/))
- **Autonomous Sleep Cycles**: When CLI is idle, [`SleepCycle.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/SleepCycle.py) triggers [`Consolidator.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/Consolidator.py) to merge short-term memory nodes into long-term structures.
- **Dream Generation**: [`DreamGenerator.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/DreamGenerator.py) synthesizes hypothetical trajectories to refine memory association weights before new prompts arrive.
- **Monte Carlo Credit Assignment**: [`CreditAssigner.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/CreditAssigner.py) updates trajectory node values and edge association strengths upon episode completion:
  $$\Delta V(S) = \alpha (G - V(S))$$

---

# 4. End-to-End Execution Lifecycle Flow

```text
                               [ Developer Prompt / CLI ]
                                   (src/input/cli.py)
                                           │
                                           ▼
                                 Workspace Context Hook
                             (src/input/hook/workspace.py)
                                           │
                                           ▼
                              Perception Observation & DTO
                             (src/input/hook/perception.py)
                                           │
                                           ▼
                              Mothership Arbitration Loop
                             (src/swarm/mothership.py)
                                           │
                       ┌───────────────────┴───────────────────┐
                       ▼                                       ▼
               Cognitive Swarm                        Cognitive Stability
             (src/swarm/cells.py)                  (Pendulum Physical Control)
                       │
                       ▼
              SwarmSAC Actor Policy ──► Latent Action Vector (2048-dim)
                       │
                       ▼
             Decoder Model Generation ──► Resilient ThoughtParser
            (SmolLM2-360M Engine)         (ThoughtParser.parse)
                       │                           │
                       ▼                           ▼
               Sandbox Execution           ScratchPad History
            (Decoder.execute_code)        (ThoughtStepDTO steps)
                       │                           │
                       └─────────────┬─────────────┘
                                     │
                                     ▼
                         Hybrid Memory Persistence
                      (ChromaDB Vector + SQLite Graph)
                                     │
                                     ▼
                         Monte Carlo Credit Assignment
                        (CreditAssigner.assign_credit)
```

---

# 5. Key Development Conventions for AI Coding Agents

When adding or modifying code in Shiva:

1. **DTO First**: Always inspect or define data structures in [`src/transferDTO.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/transferDTO.py) before creating new parameters.
2. **Parsing Safety**: Never call `json.loads()` directly on raw LLM text outputs. Always call `ThoughtParser.parse(text)` to ensure fallback safety.
3. **Repository Interfacing**: Use `HybridMemoryRepository` (or `MemoryRepository` base class) for database reads/writes rather than raw SQL calls.
4. **Sandbox Isolation**: Any Python code generated for local execution must execute via `Decoder.execute_code_in_sandbox()` and respect the `WORKSPACE_DIR` boundary.
5. **No Main Loop Blocking**: Long-running background operations (sleep/dream cycles, background model loads) must run in daemon threads or async workers so the user CLI prompt remains responsive.

---

# Technology Stack & Dependencies

- **Runtime**: Python 3.11+
- **Decoder Model**: `HuggingFaceTB/SmolLM2-360M-Instruct` (Cached under `models/shiva-decoder`)
- **Encoder Model**: `google-bert/bert-large-uncased`
- **Vector Database**: ChromaDB (HNSW Cosine Vector Store under `chroma_db/`)
- **Relational Database**: SQLite3 (WAL Mode under `memory.db`)
- **Neural & RL Framework**: PyTorch (TorchScript / MPS / CUDA / CPU)
- **Data Validation**: Pydantic V2 & Python Dataclasses

---

## Contributing
Shiva is an open-source research project. We welcome contributions in Deep Learning, Reinforcement Learning, Cognitive Science, and Software Automation. 

## License
Released under the Apache 2.0 License.

---
**Intelligence is more than prediction. It is perception, regulation, memory, adaptation, and action.**
