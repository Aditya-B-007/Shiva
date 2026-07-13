<div align="center">

# shiva.ai
### An Open-Source physical AI.

*"Building AI that doesn't just predict—it perceives, remembers, regulates, plans, and acts."*

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![Status](https://img.shields.io/badge/status-active%20development-orange.svg)
![Research](https://img.shields.io/badge/research-cognitive%20AI-purple.svg)

</div>



![shiva ai](https://github.com/user-attachments/assets/24f134a5-aae8-4044-879a-ae0c431e08b4)
# Why Shiva?

Artificial Intelligence has made remarkable progress in language understanding, reasoning, vision, and code generation. Modern Large Language Models (LLMs) can solve complex problems, write software, and interact with humans in increasingly natural ways. Despite these advances, nearly all current AI systems remain fundamentally **reactive**.

They generate responses conditioned on inputs but do not continuously perceive their environment, regulate internal cognitive processes, maintain persistent identity, or autonomously decide what deserves attention. Memory is typically implemented as retrieval over stored embeddings, planning is often prompt-driven, and reasoning begins only after a user request.

Shiva explores a fundamentally different direction.

Rather than building another language model, Shiva aims to build a **Cognitive Operating System (Cognitive OS)** capable of continuously perceiving, reasoning, remembering, planning, and acting across multiple domains.

---

# The Problem

Most AI systems today can be simplified as:

```text
Input
   │
   ▼
Large Language Model
   │
   ▼
Output
```

While powerful, this architecture assumes intelligence begins with a prompt and ends with a response.

Real-world autonomous systems do not function this way.

A robot, smartphone assistant, autonomous vehicle, or embodied AI must continuously:

- Perceive the environment.
- Maintain an internal cognitive state.
- Remember previous experiences.
- Regulate competing objectives.
- Collaborate across multiple cognitive processes.
- Decide when and how to act without explicit instructions.

The challenge is no longer building a model that answers questions.

The challenge is building software that can **continuously think**.

---

# Our Approach

Shiva models intelligence as a **distributed Cognitive Operating System** rather than a monolithic neural network.

Instead of placing a Large Language Model at the center of the architecture, Shiva decomposes cognition into specialized systems responsible for distinct aspects of intelligence.

These systems include:

- Multi-modal Perception
- Semantic Understanding
- Emotion
- Homeostasis
- Memory
- Working Memory
- Identity
- Planning
- Decision Making
- Tool Execution

Each subsystem evolves independently while cooperating through a shared cognitive runtime.

The language model is **not the brain**.

It is a **reasoning engine** used by the cognitive system whenever probabilistic inference or language generation is required.

---

# Cognitive Architecture

<img width="1051" height="881" alt="Screenshot 2026-07-11 at 11 25 46 PM" src="https://github.com/user-attachments/assets/e3c32e2a-2f01-403a-a807-db6fe3b56d0a" />

---

# Design Philosophy

Shiva is built on the belief that intelligence emerges from the interaction of specialized systems rather than from a single algorithm.

Our design principles are:

- Intelligence is a continuous process, not a prompt-response cycle.
- Cognition should be modular and independently evolvable.
- Memory must influence future behaviour.
- Emotion is an adaptive computational signal rather than a human simulation.
- Working memory and long-term memory are fundamentally different.
- Internal regulation is necessary for stable autonomous behaviour.
- Large Language Models are reasoning engines—not complete cognitive architectures.

---

# Core Concepts

### Cognitive Operating System

Shiva is designed as an operating system for cognition rather than a conversational assistant. It continuously processes events, manages internal state, schedules cognitive processes, and coordinates perception, reasoning, memory, and action.

---

### Cognitive Swarm

Instead of relying on a single monolithic controller, Shiva distributes cognition across multiple autonomous cognitive nodes.

Each node maintains its own local:

- Emotional State
- Memory Access
- Scratchpad
- Planning Process
- Cognitive Executive

Collective intelligence emerges through cooperation between these nodes.

---

### Emotion & Homeostasis

Emotion is treated as a computational mechanism that regulates:

- Attention
- Learning Priority
- Memory Formation
- Decision Bias
- Resource Allocation

Homeostasis continuously monitors internal variables such as energy, safety, stress, and cognitive load to maintain stable behaviour.

---

### Persistent Memory

Shiva separates memory into multiple layers:

- Working Memory
- Episodic Memory
- Semantic Memory
- Narrative Identity

Rather than storing every interaction, memories are consolidated based on emotional salience, novelty, and long-term utility.

---

### Shared Reasoning Engine

The reasoning engine performs probabilistic inference and language generation using open-weight language models.

Current implementation targets:

- SmolLM2

Future versions aim to support:

- Dynamic Model Migration
- Online Frankenmerging
- Specialized Reasoning Models

---

### Embodied Intelligence

Shiva is designed to operate directly on physical devices.

When installed on a smartphone, tablet, robot, or embedded platform, existing hardware becomes part of the cognitive system.

For example:

- Camera → Vision
- Microphone → Hearing
- GPS → Spatial Awareness
- Accelerometer → Balance
- Storage → Long-Term Memory
- Applications → Effectors

Rather than controlling a device through isolated API calls, Shiva treats the device itself as the body of an autonomous cognitive agent.

---

# Technology Stack

### Perception

- BERT
- Vision Encoders
- Audio Encoders

### Cognition

- Emotional Appraisal
- Homeostasis
- Memory Graphs
- Scratchpad Reasoning
- Swarm Cognition

### Reasoning

- SmolLM2
- Frankenmerged Language Models

### Reinforcement Learning

- Soft Actor-Critic (SAC)
- Prioritized Experience Replay
- Multi-Agent Reinforcement Learning

### Infrastructure

- PyTorch
- SQLite
- Plugin-Based Cognitive Runtime
- Distributed Swarm Architecture

---

# Roadmap

### Phase 1 — Foundations *(Complete)*

- Emotion Engine
- Homeostasis
- Memory System

### Phase 2 — Cognitive Runtime *(In Progress)*

- Cognitive Swarm
- Shared Reasoning Engine
- Scratchpad Coordination
- Tool Execution Framework

### Phase 3 — Embodied Intelligence

- Smartphone Runtime
- Robotics Integration
- Continuous Background Cognition
- Multi-modal Perception

### Phase 4 — Distributed Intelligence

- Multi-device Cognition
- Online Frankenmerging
- Model Migration
- Distributed Memory Synchronization

---

# Mission

> **Shiva is not another AI assistant built around a language model. It is a Cognitive Operating System that transforms existing hardware into autonomous cognitive agents capable of continuously perceiving, reasoning, remembering, planning, and acting.**


---

## Contributing
Shiva is an open-source research project. We welcome contributions in Deep Learning, Reinforcement Learning, Cognitive Science, and Robotics. 

## License
Released under the Apache 2.0 License.

---
**Intelligence is more than prediction. It is perception, regulation, memory, adaptation, and action.**
