<div align="center">

# shiva.ai
### Autonomous On-Device Cognitive Operating System for Enterprise Workspaces

*"Empowering organizations with private, sovereign, zero-cost artificial general intelligence that perceives, remembers, regulates, and executes directly within developer environments."*

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![Status](https://img.shields.io/badge/status-active%20development-orange.svg)
![Architecture](https://img.shields.io/badge/architecture-Cognitive%20Swarm-purple.svg)

</div>

---

# Executive Summary & Enterprise Value Proposition

**Shiva.ai** is a distributed, on-device Cognitive Operating System engineered for deep workspace automation, sovereign knowledge retention, and multi-agent problem solving. 

Unlike traditional cloud-hosted artificial intelligence models that act as simple prompt-response web services, Shiva operates as an integrated cognitive runtime running locally inside enterprise developer environments. It combines multi-tier neural models, continuous reinforcement learning, hybrid vector-graph long-term memory, homeostatic stability regulation, and local sandbox code execution to deliver end-to-end task completion.

---

# Enterprise Business Use Cases

### 1. Sovereign & Air-Gapped Code Generation
Enterprise software teams working on proprietary codebases, trade secrets, financial algorithms, or regulated healthcare systems cannot risk transmitting source code over external networks. Shiva runs 100% locally on workstation hardware, performing automated code refactoring, bug fixes, test generation, and architecture audits with zero data leakage.

### 2. Enterprise Persistent Memory & Institutional Knowledge Retention
Standard language models suffer from catastrophic forgetting once a chat session terminates. Shiva continuously indexes developer actions, project commits, architectural choices, and execution outcomes into a hybrid vector-graph memory store. Institutional knowledge is permanently retained, reinforced, and retrieved across teams without external database costs.

### 3. Autonomous 24/7 Sleep Consolidation & Synthetic Dreaming
During developer downtime, Shiva enters an automated background sleep cycle. It prunes dormant memory nodes, consolidates short-term scratchpad trajectories into long-term knowledge, and runs synthetic dream simulations to optimize future search efficiency—ensuring workstations wake up smarter every morning.

### 4. Regulated Multi-Agent Swarm Engineering
Complex software challenges require multiple specialized perspectives. Shiva orchestrates a swarm of dedicated Cortical Columns (Analytical, Creative, Risk & Safety, and Verification) under an executive prefrontal controller. Dynamic physical stability regulation prevents cognitive loops, hallucinations, and runaway operations.

---

# Strategic Advantages over Proprietary Closed Cloud AI Models

Shiva was built to address the critical architectural, financial, and security limitations inherent in proprietary closed cloud LLM APIs and legacy metered assistant services.

| Feature / Capability | Proprietary Closed Cloud AI APIs | **Shiva.ai (On-Device Cognitive OS)** |
| :--- | :--- | :--- |
| **Data Privacy & Sovereignty** | **High Risk**: Source code, intellectual property, and internal context are transmitted to remote third-party servers. | **100% Sovereign**: Runs fully on-device. Zero network requests, zero third-party logging, air-gap compliant. |
| **Operational & Token Costs** | **Unbounded Cost**: Metered per-token pricing scales exponentially as context grows across large codebases. | **$0 Ongoing API Cost**: Utilizes local hardware compute with zero per-token or subscription fees. |
| **Memory Lifetime & Context** | **Stateless**: Context is lost when prompt windows fill up or chat sessions are closed. | **Infinite Hybrid RAG Memory**: Dual ChromaDB vector store & SQLite relational graph topology persists indefinitely across sessions. |
| **System Reliability & Crash Safety** | **Fragile**: Malformed model outputs or API timeouts disrupt automated developer pipelines. | **Zero-Crash Resilient Pipeline**: 5-tier fallback cascade (`ThoughtParser`) guarantees structured execution under all conditions. |
| **Self-Regulation & Hallucinations** | **Unregulated**: Cloud models lack physical self-awareness, frequently generating unverified hallucinated code. | **Physical Pendulum Regulation**: Prefrontal Mothership uses inverted pendulum dynamics to measure uncertainty and trigger risk columns. |
| **Reinforcement & Self-Improvement** | **Static**: Closed weights remain static; model cannot learn from daily workspace actions. | **Continuous RL (Swarm SAC)**: Soft Actor-Critic policy continuously reinforces successful actions via temporal difference learning. |
| **Execution Environment** | **Constrained**: Relies on remote cloud proxies or manual user copy-pasting of code snippets. | **Local Execution Sandbox**: Direct, sandboxed Python runtime execution with automated unit test verification. |

---

## Comparative Value Matrix

### 1. Eliminating Intellectual Property Exposure
Proprietary cloud platforms require sending confidential codebases to centralized servers, exposing organizations to compliance breaches, data leaks, and unauthorized training usage. Shiva provides complete operational privacy by keeping all weights, vector embeddings, relational graphs, and code sandboxes strictly on local corporate hardware.

### 2. Zero-Cost Scaling for Heavy Developers
As developer prompts and project contexts expand, metered API billing for proprietary closed models creates unpredictable overhead. Shiva leverages local accelerator hardware (CUDA / Apple Silicon MPS / CPU), delivering unlimited reasoning iterations, multi-agent swarm arbitration, and continuous background sleep cycles at zero incremental cost.

### 3. Deterministic Governance vs. Unpredictable LLM Outputs
Proprietary web endpoints regularly break structured downstream parsing due to unexpected format shifts or unclosed tags. Shiva enforces a zero-crash architecture: every model output passes through a 5-stage parsing pipeline that automatically extracts structured JSON, cleans syntax errors, and falls back to safe operational defaults if necessary.

---

# Architecture Overview & Developer Navigation

For a detailed, file-by-file and line-by-line technical breakdown of the entire Shiva repository for AI agents and human developers, refer to the dedicated agent knowledge base:

- 📑 [Architecture Overview](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/00_ARCHITECTURE_OVERVIEW.md)
- 📑 [Data Transfer Objects (`src/transferDTO.py`)](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/01_TRANSFER_DTO.md)
- 📑 [Input, Perception & Output Subsystems](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/02_INPUT_OUTPUT_PACKAGING.md)
- 📑 [Cognitive Transformer & Single-Node Reasoning](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/03_BRAIN_TRANSFORMER_NODE.md)
- 📑 [Multi-Tier Memory Architecture](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/04_BRAIN_MEMORY.md)
- 📑 [Regulation, Transactional Gates & Infrastructure](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/05_BRAIN_EMOTION_GATE_INFRASTRUCTURE.md)
- 📑 [Distributed Swarm Intelligence & RL Policy](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/md_file_for_agent/06_SWARM_SUBSYSTEM.md)

---

# Quick Start & Installation

### Requirements
- Python 3.11+
- PyTorch with CUDA or Apple Silicon MPS support (optional, CPU supported)

### Installation
```bash
# Clone the repository
git clone https://github.com/shiva-ai/shiva.git
cd shiva

# Install dependencies
pip install -r src/packaging/requirements.txt
```

### Running Shiva CLI
```bash
# Submit a single problem query
python -m src.input.cli -q "Analyze the current workspace and refactor bug in module" -d ./

# Interactive terminal session
python -m src.input.cli -i -d ./

# Enable Voice Output synthesis (WAV output)
python -m src.input.cli -q "Summarize repository architecture" -v
```

---

# License

Distributed under the Apache 2.0 License. See `LICENSE` for details.
