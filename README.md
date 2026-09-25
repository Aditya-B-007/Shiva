<div align="center">

# 🧠 Sankalpa

**A 110M-parameter decision intelligence engine — not a chatbot, a choice-maker.**

[![Pure NumPy](https://img.shields.io/badge/Built%20With-Pure%20NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Parameters](https://img.shields.io/badge/Parameters-110M-blueviolet?style=for-the-badge)](https://github.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)
[![Stage](https://img.shields.io/badge/Stage-Research-f59e0b?style=for-the-badge)](https://github.com)
[![Memory](https://img.shields.io/badge/RAM%20(INT8)-110%20MB-ef4444?style=for-the-badge)](https://github.com)
[![Framework](https://img.shields.io/badge/Framework-None%20Required-64748b?style=for-the-badge)](https://github.com)

</div>

---

## What Is Sankalpa?

Most AI models answer questions by generating text. **Sankalpa does something different** — it reads a situation and ranks a list of candidate actions by probability, returning a *precise, auditable confidence score for each option*.

Feed it a scenario. Give it your options. Get back a ranked probability distribution — no hallucination, no freeform output, no guesswork.

```
Input:   "Patient is a 58-year-old male with acute chest pain radiating to left arm,
          BP 85/50, heart rate 115, ST elevation in leads II, III, aVF."

Options: ["Administer IV Beta-Blockers",
          "Immediate Cardiac Catheterization Lab Activation",
          "Discharge with pain medication",
          "Observe in waiting room for 4 hours"]

Output:  🏆 Immediate Cardiac Catheterization Lab Activation  —  94.2% confidence
         2. Administer IV Beta-Blockers                       —   4.1%
         3. Observe in waiting room for 4 hours               —   1.4%
         4. Discharge with pain medication                    —   0.3%
```

---

## The Problem It Solves

Large language models are designed for conversation. When you force them into structured decision-making, you get three problems:

| Problem | What happens in practice |
|---|---|
| **Unpredictability** | The same prompt gives different answers across runs |
| **Opacity** | No probability scores — just words you have to interpret |
| **Cost & Latency** | Billions of parameters to answer a yes/no question |
| **Auditability** | "The AI said so" is not a defensible answer in regulated industries |

Sankalpa is built specifically for the **structured decision layer** — the moment where a system must choose between a finite set of known actions, and that choice must be fast, consistent, and traceable.

---

## Industry Use Cases

| Vertical | Scenario | What Sankalpa does |
|---|---|---|
| 🏥 **Healthcare** | Critical care triage with vitals, labs, imaging | Ranks interventions by clinical urgency |
| 🖥️ **DevOps / SRE** | Server temperature spike, DB latency alert, fan failure | Selects automated remediation action |
| 🚁 **Autonomous Systems** | Drone battery at 6%, headwind 35kn, base 4.2km away | Chooses safest action given constraints |
| 🤖 **Industrial Robotics** | Sudden force spike on Z-axis during precision insertion | Selects compliant motion recovery strategy |
| 💳 **Financial Security** | 500 identical withdrawals from 500 IPs in 200ms | Flags fraud vs. processes vs. throttles |
| ⚖️ **Legal / Compliance** | Contract clause conflict with regulatory requirement | Ranks resolution paths by risk exposure |

---

## Why Choose Sankalpa

### ✅ Deterministic by Design
Set `seed=42` and the model produces **bit-identical output** on every run. No random sampling, no temperature scaling surprises — the same situation always maps to the same probability distribution.

### ✅ Fully Auditable
Every decision is a matrix multiply you can inspect. The full `[1, K]` probability vector, logit scores, and cosine similarities are returned alongside the winner — not just a label.

### ✅ Extremely Lightweight
| Precision | Memory |
|---|---|
| FP32 | ~441 MB |
| FP16 / BF16 | ~220 MB |
| INT8 | **~110 MB** |

Runs on CPU. No GPU required. No CUDA, no Docker, no cloud dependency.

### ✅ Zero Framework Lock-In
Built entirely in **pure NumPy**. No PyTorch, no TensorFlow, no ONNX runtime. Every weight matrix, attention head, and backpropagation step is explicit and readable Python.

### ✅ Fine-Tunable on Your Data
Two-stage training pipeline included:
- **Stage 1 — DPO:** Teach the model your preferences with `(scenario, preferred_action, rejected_action)` pairs
- **Stage 2 — RLCD:** Calibrate confidence scores to match your domain's risk profile

Add 10 labelled examples and retrain in minutes.

### ✅ Inference-Time Reasoning
Enable `thinking mode` to activate **Monte Carlo Tree Search (MCTS)** — the model runs 64 simulated decision paths using a PUCT policy before committing to a final answer.

---

## Quickstart

```bash
# Clone and enter the project
git clone <repo-url> && cd Shiva

# Install dependencies
pip install numpy pydantic scikit-learn PyPDF2

# Run end-to-end training + inference validation
python3 training/trainingScript.py
```

Training will automatically load the saved tokenizer from `model_artifacts/tokenizer/tokenizer.json` if it exists, then train both DPO and RLCD stages, and print a ranked decision for a held-out test scenario.

---

<br>

<div align="center">

---

## 🔬 Technical Architecture

*The following section is for engineers and researchers. It contains the full parameter breakdown, mathematical formulation, and memory analysis.*

---

</div>

## 🧠 110M Architecture Breakdown

This section provides a layer-by-layer mathematical and structural summary of the 110 Million parameter model architecture, detailing how input is processed, reasoning occurs, and decisions are made.

---

## 📊 Parameter Summary Table

| Component | Layer / Operation | Dimensions & Math | Parameter Count (Approx.) |
| :--- | :--- | :--- | :--- |
| **1. Embedding Layer** | Word Token Embeddings | $30{,}522 \text{ words} \times 768$ | $23.44$ Million |
| | Position Embeddings | $512 \text{ positions} \times 768$ | $393{,}216$ |
| | Token Type (Segment) | $2 \text{ types} \times 768$ | $1{,}536$ |
| | Embedding LayerNorm | $2 \times 768$ (scale + shift) | $1{,}536$ |
| **2. 12-Layer Encoder** | Self-Attention ($Q, K, V$) | $12 \text{ layers} \times 3 \times (768^2 + 768)$ | $21.26$ Million |
| | Attention Output Dense | $12 \text{ layers} \times (768^2 + 768)$ | $7.09$ Million |
| | Attention LayerNorm | $12 \text{ layers} \times (2 \times 768)$ | $18{,}432$ |
| | Feed-Forward Layer 1 | $12 \text{ layers} \times (768 \cdot 3072 + 3072)$ | $28.35$ Million |
| | Feed-Forward Layer 2 | $12 \text{ layers} \times (3072 \cdot 768 + 768)$ | $28.32$ Million |
| | Feed-Forward LayerNorm | $12 \text{ layers} \times (2 \times 768)$ | $18{,}432$ |
| **3. Situation MLP** | Compression Dense 1 | $768 \times 512 + 512$ | $393{,}728$ |
| | Intermediate LayerNorm | $2 \times 512$ | $1{,}024$ |
| | Refinement Dense 2 | $512 \times 512 + 512$ | $262{,}656$ |
| | Final Situation LayerNorm | $2 \times 512$ | $1{,}024$ |
| **4. Option Converter** | Shared Encoder Weights | Re-uses Component 1 & 2 weights | $0$ (Tied) |
| | Option Projector MLP | $768 \times 512 + 512$ | $393{,}728$ |
| **5. Final Head** | Bilinear Alignment ($W$) | $512 \times 512$ matrix | $262{,}144$ |
| | Learnable Temperature ($\tau$) | $1 \text{ scalar}$ | $1$ |
| | Parallel MatMul + Softmax | Deterministic Linear Algebra | $0$ (Pure Math) |
| **TOTAL** | | | $\mathbf{\approx 110.2 \text{ Million}}$ |

---

## ⚙️ Detailed Mathematical Walkthrough

### 1. Input Embeddings ($\approx 23.8 \text{M Parameters}$)
The goal is to convert raw token IDs into dense, position-aware vectors.

*   **Token Embeddings:** A lookup table mapping vocabulary tokens to 768-dimensional vectors.
    $$\text{Parameters} = 30{,}522 \times 768 = 23{,}440{,}896$$
*   **Position Embeddings:** Provides sequential awareness across the context window.
    $$\text{Parameters} = 512 \times 768 = 393{,}216$$
*   **Segment Embeddings & LayerNorm:** Distinguishes context/query segments and applies normalization.
    $$\text{Parameters} = (2 \times 768) + (2 \times 768) = 3{,}072$$

### 2. The 12-Layer Transformer Backbone ($\approx 85.0 \text{M Parameters}$)
This is the core language understanding engine, utilizing self-attention and feed-forward networks.

*   **Self-Attention Sub-Layer:** For each layer, this computes Query ($Q$), Key ($K$), and Value ($V$) projections.
    $$\text{Parameters per layer} = 3 \times (768^2 + 768) = 1{,}771{,}776$$
*   **Attention Output Dense:** Combines the attention outputs across all heads.
    $$\text{Parameters per layer} = (768^2 + 768) = 590{,}592$$
*   **LayerNorm:** Stabilizes activations within the layer.
    $$\text{Parameters per layer} = 2 \times 768 = 1{,}536$$
*   **Feed-Forward Network (FFN):** This expands and then contracts the feature representation.
    *   **Layer 1 (Expansion):** $768 \rightarrow 3{,}072$
        $$\text{Parameters per layer} = (768 \times 3{,}072 + 3{,}072) = 2{,}362{,}368$$
    *   **Layer 2 (Projection):** $3{,}072 \rightarrow 768$
        $$\text{Parameters per layer} = (3{,}072 \times 768 + 768) = 2{,}360{,}064$$

**Total Calculation for 12 Layers:**
$$\text{Total per layer} = 1{,}771{,}776 + 590{,}592 + 1{,}536 + 2{,}362{,}368 + 2{,}360{,}064 + 1{,}536 \approx 7{,}087{,}872$$
$$\mathbf{\text{Total for 12 layers}} = 12 \times 7{,}087{,}872 \approx \mathbf{85.05 \text{ Million}}$$

### 3. Situation MLP ($\approx 0.66 \text{M Parameters}$)
This network compresses the rich text representation into a concise, fixed-size state vector ($z$).

*   **Compression Dense 1:**
    $$\text{Parameters} = (768 \times 512) + 512 = 393{,}728$$
*   **Intermediate LayerNorm:**
    $$\text{Parameters} = 2 \times 512 = 1{,}024$$
*   **Refinement Dense 2:**
    $$\text{Parameters} = (512 \times 512) + 512 = 262{,}656$$
*   **Final LayerNorm:**
    $$\text{Parameters} = 2 \times 512 = 1{,}024$$

### 4. Option Matrix Converter ($\approx 0.39 \text{M Parameters}$)
Candidate options are projected into the same vector space as the situation vector by re-using the core encoder components, minimising distributional mismatch.

*   **Option Projector MLP:**
    $$\text{Parameters} = (768 \times 512) + 512 = 393{,}728$$

### 5. Final Head ($\approx 0.26 \text{M Parameters}$)
Computes compatibility scores between the state vector and all options simultaneously.

*   **Bilinear Alignment Matrix ($W$):** Captures the feature trade-offs between situation and options.
    $$\text{Parameters} = 512 \times 512 = 262{,}144$$
*   **Learnable Temperature ($\tau$):** A single scalar that scales the final probability distribution.
    $$\text{Parameters} = 1$$
*   **Parallel Matrix Multiplication:** Core computation — matrix product followed by Softmax.
    $$\text{Parameters} = 0 \text{ (Pure Math)}$$

---

## 💾 Memory Footprint Analysis

| Precision | Bytes/Param | Total Memory |
|---|---|---|
| FP32 | 4 | **~440.8 MB** |
| FP16 / BF16 | 2 | **~220.4 MB** |
| INT8 | 1 | **~110.2 MB** |

This profile is optimised for edge deployment — fitting comfortably within modest consumer hardware or embedded systems with ample headroom for batch processing.

---

## 🔢 Understanding the Numbers

**768** is **not** the context length — it is the **feature width (hidden dimension)**.

**512** appears in two places: once as the **maximum context length** (token sequence length), and once as the **decision space width** (the length of the Situation Vector $z$).

### The Matrix Mental Model

Think of the data passing through the Transformer as a 2D grid:

$$\text{Shape: } [\text{Sequence Length} \times \text{Hidden Dimension}]$$

* **Rows (Sequence Length):** How many words/tokens are in the sentence — up to **512** tokens.
* **Columns (Hidden Dimension):** How many numbers describe the meaning of *one single word* — **768** numbers.

### Breakdown of Every Number

| Number | Name | Role |
|---|---|---|
| **768** | Hidden Dimension $d_{\text{model}}$ | Width of every token's feature vector through all 12 layers |
| **512** (context) | Maximum Context Length | Max tokens the encoder reads at once |
| **512** (decision) | Situation Vector $z \in \mathbb{R}^{512}$ | Compressed decision state after FinalMLP |
| **3,072** | FFN Expansion Dimension | Temporary 4× width inside each FFN ($768 \times 4$) |
| **12** (layers) | Encoder Depth | Stacked Transformer encoder blocks |
| **12** (heads) | Attention Heads | Parallel contextual relationships per layer |
| **64** | Head Dimension $d_k$ | Q/K/V vector size per head ($768 / 12$) |
| **30,522** | Vocabulary Size | Unique word and sub-word pieces in the BPE tokenizer |
| **K** | Candidate Options Count | User-supplied choices, forming Options Matrix $[K \times 512]$ |
