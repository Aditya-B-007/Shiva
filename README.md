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

Shiva is a command-line cognitive human like-agent which acts as an integrated cognitive model rather than just a simple wrapper around an AI model. In a nutshell it is a general intellligence. Directly rivals the General Intelligence Model of China.

Rather than building another conversational chatbot, Shiva aims to build a Cognitive General Intelligence model capable of continuously planning, reasoning, executing code, remembering, and adapting directly inside developer workspaces.

---

# The Problem

Most workspace AI tools today are purely reactive, the agents are robotic, relying on probabilistic ways of coding and generating the necessary language. As a founder and CTO, would you like your coding agent or email drafting agent to be robotic or like a person who does all of it, with no data being sent to external servers, the person is a software. Yes correct !! This is why we are building shiva.ai

Intelligence should not begin with a prompt and end with a response. Real-world autonomous General Intelligence systems must continuously think, reason, remember previous executions, self-critique, and execute sandbox operations without manual prompt-chaining.

---

# Our Approach

Shiva models workspace automation as a **distributed Cognitive Operating System** rather than a monolithic neural network. It decomposes cognition into specialized subsystems responsible for distinct aspects of intelligence, coordinating them through a shared runtime.

The language model is **not the brain**. It is a **reasoning engine** used by the cognitive system whenever probabilistic inference, planning, or code generation is required.

---

# Cognitive Architecture

```text
                 [ Dev Workspace / CLI ]
                           │
                           ▼
                  Workspace Context Hook
                 (Grep / Directory Scans)
                           │
                           ▼
               Semantic Understanding & Memory
                           │
                           ▼
               Mothership Arbitration Loop
                           │
           ┌───────────────┴───────────────┐
           ▼                               ▼
       Cognitive Swarm             Internal Regulation    
           │
           ▼
    Chain of Thought
  (Thought • Critique)
           │
           ▼
   Sandbox Execution 
           │
           ▼
  Monte Carlo Assignment ─────── SQLite MemoryGraph
```

---

# Design Philosophy

- **Unified DTOs**: Clean data tracking using a consolidated DTO model in `src/transferDTO.py`.
- **Integrated Sandbox Execution**: Reasoning is compiled into executable Python scripts, run in a secure local sandbox with rich standard libraries, and results are appended directly back to the decision chain.
- **Autonomous Idle Processing**: When the CLI is waiting for user prompts, Shiva automatically runs sleep and dream consolidation cycles in the background, suspending instantly on user input.
- **Monte Carlo Credit Assignment**: At the end of each reasoning episode, trajectory steps are evaluated and reinforced in a vector-less SQLite MemoryGraph.

---

# Core Concepts

### 1. Chain of Thought + Sandbox Execution
Shiva columns generate multi-turn reflections, critiques, and decisions. If a column outputs a Python 3 block to read/write files or execute calculations, the `Decoder` executes it in the local sandbox under a `WORKSPACE_DIR` context, returning standard outputs back to the decision flow.

### 2. Autonomous Idle Sleep & Dream Cycles
When idle, Shiva runs background sleep threads utilizing `DreamGenerator` and the consolidation/forgetting/identity models. As soon as the user enters a prompt, the dream thread immediately stops, commits memory states to SQLite, and executes the query.

### 3. Reinforcement Credit Assignment
Shiva resolves multi-step credit assignment using Monte Carlo estimation. Trajectory node paths and temporal edges are mathematically reinforced:
$$\Delta V(S) = \alpha (G - V(S))$$
This updates memory `strength` and `association_strength` inside the SQLite memory repository based on the final query confidence.

###4. Mothership and cells
Remember "The Avengers" film, battle of new york, where an army of chitauri had come in. the two legged creatures were being launched by a big flying whale. shiva.ai functions the same way were it as a mothership which deployes cells for solving a problem. Each cell has its own strength.

---

# Technology Stack

### Cognition
- **Memory**: Custom vector-less SQLite `MemoryGraph` with topological activation spreading.
- **Regulation**: Prefrontal `Mothership` coordinator and physical pendulum stability regulator.
- **Appraisal**: FT-Transformer-based emotional dynamics appraisal engine.

### Language used: python 3.11

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
