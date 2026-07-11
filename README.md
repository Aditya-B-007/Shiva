<div align="center">

# shiva.ai
### An Open-Source Cognitive Operating System for Autonomous AI Systems

*"Building AI that doesn't just predict—it perceives, remembers, regulates, plans, and acts."*

![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-green.svg)
![Status](https://img.shields.io/badge/status-active%20development-orange.svg)
![Research](https://img.shields.io/badge/research-cognitive%20AI-purple.svg)

</div>

---

![shiva ai](https://github.com/user-attachments/assets/24f134a5-aae8-4044-879a-ae0c431e08b4)

## 🎯 The Core Concept

Shiva is a **Cognitive application** designed to convert any device (smartphone, computer, or embedded platform) into a robot, executing tasks autonomously as if it were a human. 

Instead of treating intelligence as a simple, reactive question-and-answer cycle, Shiva decomposes cognition into continuous, modular processes—such as multi-modal perception, homeostatic regulation, emotional dynamics, and persistent memory graphs—coordinated by an active reasoning loop.

---

## ⚡ The Problem: Reactive AI vs. Autonomous Agency

Modern AI systems are fundamentally **reactive**. They sit idle, waiting for a prompt, and immediately shut down after returning a text response:

```text
User Input ──▶ [ Large Language Model ] ──▶ Text Output (Context Forgotten)
```

This model is insufficient for true autonomous agency because:
* **No Continuous Awareness**: The agent cannot sense changes in its environment (battery status, network quality, sensor signals) unless explicitly prompted.
* **No Internal Drive**: There are no emotional states or homeostatic metrics (energy, load, stress) to guide resource allocation and attention.
* **Transient Memory**: Retrievals are limited to stateless vector searches rather than associative, persistent memory graphs.

---

## 🛡️ The Solution: The Cognitive Operating System

Shiva resolves this by treating the reasoning model not as the "brain," but as a **generative reasoning engine** called by an active, continuous cognitive loop:

<div align="center">

### Interactive Live Data-Flow

<svg width="600" height="420" viewBox="0 0 600 420" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Background Glow & Grid -->
  <rect width="600" height="420" rx="12" fill="#0d1117"/>
  <defs>
    <linearGradient id="glowGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.2"/>
      <stop offset="100%" stop-color="#a855f7" stop-opacity="0.1"/>
    </linearGradient>
    <filter id="glowFilter" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="6" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>
  <rect width="600" height="420" rx="12" fill="url(#glowGrad)"/>

  <!-- Connections / Lines -->
  <!-- Perception -> Encoder -->
  <path d="M 120 210 L 220 210" stroke="#4b5563" stroke-width="2" stroke-dasharray="4 4" />
  <!-- Encoder -> Swarm Nodes -->
  <path d="M 300 210 L 380 150" stroke="#4b5563" stroke-width="2" />
  <path d="M 300 210 L 380 270" stroke="#4b5563" stroke-width="2" />
  <!-- Swarm Nodes -> Memory & Emotion -->
  <path d="M 460 150 L 520 210" stroke="#4b5563" stroke-width="2" />
  <path d="M 460 270 L 520 210" stroke="#4b5563" stroke-width="2" />
  <!-- Feedback Loop -->
  <path d="M 520 210 C 520 370, 120 370, 120 240" stroke="#38bdf8" stroke-dasharray="5 5" stroke-width="1.5">
    <animate attributeName="stroke-dashoffset" values="50;0" dur="4s" repeatCount="indefinite" />
  </path>

  <!-- Signal Pulse Animation -->
  <circle r="4" fill="#38bdf8" filter="url(#glowFilter)">
    <animateMotion dur="3s" repeatCount="indefinite" path="M 120 210 L 220 210" />
  </circle>
  <circle r="4" fill="#a855f7" filter="url(#glowFilter)">
    <animateMotion dur="3s" repeatCount="indefinite" path="M 300 210 L 380 150" />
  </circle>
  <circle r="4" fill="#ec4899" filter="url(#glowFilter)">
    <animateMotion dur="3s" repeatCount="indefinite" path="M 300 210 L 380 270" />
  </circle>

  <!-- Nodes (Components) -->
  <!-- Multi-modal input -->
  <g transform="translate(120, 210)">
    <circle r="36" fill="#1f2937" stroke="#38bdf8" stroke-width="2" />
    <circle r="42" fill="none" stroke="#38bdf8" stroke-width="1" stroke-opacity="0.3">
      <animate attributeName="r" values="36;46;36" dur="2s" repeatCount="indefinite" />
    </circle>
    <text fill="#e5e7eb" font-family="sans-serif" font-size="10" font-weight="bold" text-anchor="middle" y="-2">PERCEPTION</text>
    <text fill="#9ca3af" font-family="sans-serif" font-size="8" text-anchor="middle" y="10">Sensory Inputs</text>
  </g>

  <!-- Semantic Processing -->
  <g transform="translate(260, 210)">
    <rect x="-40" y="-25" width="80" height="50" rx="6" fill="#1f2937" stroke="#a855f7" stroke-width="2" />
    <text fill="#e5e7eb" font-family="sans-serif" font-size="11" font-weight="bold" text-anchor="middle" y="2">ENCODER</text>
    <text fill="#c084fc" font-family="sans-serif" font-size="8" text-anchor="middle" y="14">Semantic Model</text>
  </g>

  <!-- Swarm Node 1 -->
  <g transform="translate(420, 150)">
    <rect x="-40" y="-20" width="80" height="40" rx="4" fill="#1f2937" stroke="#e2e8f0" stroke-width="1.5" />
    <text fill="#f3f4f6" font-family="sans-serif" font-size="10" text-anchor="middle" y="4">Cognitive Node</text>
    <text fill="#9ca3af" font-family="sans-serif" font-size="8" text-anchor="middle" y="14">Scratchpad</text>
  </g>

  <!-- Swarm Node 2 -->
  <g transform="translate(420, 270)">
    <rect x="-40" y="-20" width="80" height="40" rx="4" fill="#1f2937" stroke="#e2e8f0" stroke-width="1.5" />
    <text fill="#f3f4f6" font-family="sans-serif" font-size="10" text-anchor="middle" y="4">Reasoning Loop</text>
    <text fill="#9ca3af" font-family="sans-serif" font-size="8" text-anchor="middle" y="14">CoT Steps</text>
  </g>

  <!-- Core Subsystems -->
  <g transform="translate(520, 210)">
    <circle r="36" fill="#1f2937" stroke="#ec4899" stroke-width="2" />
    <circle r="42" fill="none" stroke="#ec4899" stroke-width="1" stroke-opacity="0.3">
      <animate attributeName="r" values="36;46;36" dur="3s" repeatCount="indefinite" />
    </circle>
    <text fill="#e5e7eb" font-family="sans-serif" font-size="10" font-weight="bold" text-anchor="middle" y="-2">DECODER</text>
    <text fill="#f472b6" font-family="sans-serif" font-size="8" text-anchor="middle" y="10">Reasoning Engine</text>
  </g>
</svg>

</div>

---

## 🛠️ Architecture and Subsystems

Shiva decomposes reasoning and autonomous regulation into independent subsystems:

1. **Continuous Perception**: Transforms environmental changes (device energy, sensor parameters) into embedding vectors using standard **Encoder** models.
2. **Emotional & Homeostatic Regulation**: An appraisal engine regulates utility thresholds, dynamically updating the agent's attention priorities based on computational stress, battery load, and cognitive fatigue.
3. **Cognitive Swarm & Scratchpad**: Multiple cognitive nodes process tasks concurrently. Each node maintains a local, temporary `ScratchPad` (working memory) and guides the iterative loop using a pluggable `ChainOfThought` controller.
4. **Shared Reasoning Engine**: When complex generation is required, nodes schedule and execute time-sliced generations through the shared **Decoder** model.
5. **Associative Graph Memory**: Stores long-term semantic and episodic connections as a living relationship graph, avoiding flat vector lookups.
6. **Ability to execute actions**: Using the sensors of the device, it can think and then execute actions. The LLM is just used for selecting the right tool for executing the action.

---

## 🚀 Technology Stack

* **Perception/Encoding**: PyTorch, Hugging Face Transformers
* **Decoder/Reasoning**: Hugging Face CausalLM
* **State & Memory Database**: SQLite, NetworkX-driven Graph Repositories
* **Cognitive Runtime**: Concurrent Thread-Pool schedulers with resource locks

---

## 📅 Roadmap

- **Phase 1 — Foundations** *(Completed)*: Homeostasis model, emotional appraisal engines, memory persistence graphs.
- **Phase 2 — Cognitive Runtime** *(In Progress)*: Swarm scheduling engine, DTO-wrapped reasoning loops, shared decoder resource locking.
- **Phase 3 — Embodied Agents**: Robotics execution wrappers, smartphone background cognitive runtimes.
- **Phase 4 — Action Execution Systems**: Ability to execute actions like replying back, or using vibrations in a smartphone to make small movements, take a photo, control files etc.

---

## 🤝 Contributing & Research

We welcome contributions in Deep Learning, Reinforcement Learning, Cognitive Science, and Robotics. Shiva is released under the **Apache 2.0 License**.

---

<img width="1063" height="874" alt="Screenshot 2026-07-11 at 5 52 18 PM" src="https://github.com/user-attachments/assets/cfa2c0ba-59d0-4cf4-85e2-cd883b86709d" />
