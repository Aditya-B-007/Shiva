# Architectural Overview & Agent Reference Manual

Welcome to **Shiva.ai** — an open-source, on-device Cognitive Operating System designed for general intelligence and workspace autonomy.

This documentation suite inside `md_file_for_agent/` is structured to provide any AI agent or software engineer with an immediate, complete, and line-by-line understanding of every file, data transfer object, cognitive algorithm, neural model hook, and execution loop in the repository.

---

## Subsystem Navigation Map

The codebase is organized into modular cognitive layers:

| Subsystem Module | Target Documentation File | Primary File Locations | Key Responsibilities |
| :--- | :--- | :--- | :--- |
| **Data Transfer Objects** | [`01_TRANSFER_DTO.md`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/01_TRANSFER_DTO.md) | `src/transferDTO.py` | Single source of truth for all Pydantic & dataclass interfaces across perception, memory, thought parsing, and swarm coordination. |
| **Input / Perception & I/O** | [`02_INPUT_OUTPUT_PACKAGING.md`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/02_INPUT_OUTPUT_PACKAGING.md) | `src/input/`, `src/output/`, `src/packaging/` | Terminal CLI loop, workspace directory sensing, file sandbox operations, voice synthesis (Kokoro TTS), and PyInstaller standalone build configuration. |
| **Cognitive Transformer & Node** | [`03_BRAIN_TRANSFORMER_NODE.md`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/03_BRAIN_TRANSFORMER_NODE.md) | `src/brain/transformer/`, `src/brain/node/` | Local SmolLM2-360M decoder, BERT-large encoder, 5-stage resilient thought parser, working memory Scratchpad, and ChainOfThought loop. |
| **Hybrid Memory System** | [`04_BRAIN_MEMORY.md`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/04_BRAIN_MEMORY.md) | `src/brain/memory/` | Hybrid RAG scoring (vector cosine + keyword stem + graph activation), ChromaDB vector store, SQLite relational graph topology, credit assignment, and sleep consolidation cycles. |
| **Regulation, Gates & Infra** | [`05_BRAIN_EMOTION_GATE_INFRASTRUCTURE.md`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/05_BRAIN_EMOTION_GATE_INFRASTRUCTURE.md) | `src/brain/emotionalHandlerAndStore/`, `src/brain/gate/`, `src/brain/infrastructure/` | FT-Transformer appraisal, homeostasis (energy, focus, stress), transactional memory write gates, promotion policies, thread-safe SQLite connection managers, and async queues. |
| **Swarm Intelligence & RL** | [`06_SWARM_SUBSYSTEM.md`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/06_SWARM_SUBSYSTEM.md) | `src/swarm/` | Prefrontal Mothership executive control, physical inverted pendulum stability regulator, specialized Cortical Columns (Analytical, Creative, Risk, Verification), and Soft Actor-Critic (SAC) continuous RL policy. |

---

## Core Execution & Cognitive Cycle

```
[ User Request / CLI / Voice WS ]
              │
              ▼
    [ Perception & Workspace Context ] ─── (src/input/hook/workspace.py & perception.py)
              │
              ▼
   [ Swarm Mothership Executive ] ─────── (src/swarm/mothership.py)
              │
              ├──> [ Inverted Pendulum Stability Regulator ] ── (Theta Instability Tracking)
              │
              ├──> [ Arbitrate Cortical Columns ] ───────────── (Analytical, Creative, Risk, Verification)
              │           │
              │           ▼
              │     [ Node Processing Engine ] ──────────────── (src/brain/node/nodeProcessingEngine.py)
              │           │
              │           ├──> [ Retrieve Hybrid RAG ] ──────── (src/brain/memory/MemoryEngine.py)
              │           ├──> [ Evaluate Emotion & Stress ] ── (src/brain/emotionalHandlerAndStore/)
              │           ├──> [ Generate Decision ] ────────── (src/brain/transformer/Decoder.py & thought_parser.py)
              │           └──> [ Sandbox Python Execution ] ── (Local Code Compilation & Validation)
              │
              ├──> [ Soft Actor-Critic RL Update ] ──────────── (src/swarm/SwarmSAC.py)
              │
              ▼
   [ Final Solution & Synthesis ] ─────── (MothershipResponseDTO -> Voice Output / Terminal Output)
```

---

## Global Design Invariants

1. **Zero-Crash Resilient Parsing**: All LLM text streams pass through `ThoughtParser.parse()`, enforcing structured fallback cascades to prevent formatting errors from crashing runtime loops.
2. **Deterministic Fallback Capability**: System automatically detects missing local ML packages or weights and degrades gracefully to deterministic CLI fallback handlers.
3. **Local Sovereignty**: All models (`SmolLM2-360M`, `bert-large-uncased`, `Kokoro TTS`), databases (`ChromaDB`, `SQLite`), and sandboxes run 100% locally on device without external network API calls.
4. **Physical Homeostatic Feedback**: Prefrontal arbitration is governed by dynamic equations translating cognitive uncertainty and stress into mechanical control inputs.
