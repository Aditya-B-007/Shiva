Sankalpa is a model that takes in data and outputs probability and single worded decisions not text output.

# 🧠 110M Architecture Breakdown

This document provides a layer-by-layer mathematical and structural summary of the 110 Million parameter AI model architecture, detailing how input is processed, reasoning occurs, and decisions are made.

---

## 📊 Parameter Summary Table

| Component | Layer / Operation | Dimensions & Math | Parameter Count (Approx.) |
| :--- | :--- | :--- | :--- |
| **1. Embedding Layer** | Word Token Embeddings | $30,522 \text{ words} \times 768$ | $23.44$ Million |
| | Position Embeddings | $512 \text{ positions} \times 768$ | $393,216$ |
| | Token Type (Segment) | $2 \text{ types} \times 768$ | $1,536$ |
| | Embedding LayerNorm | $2 \times 768$ (scale + shift) | $1,536$ |
| **2. 12-Layer Encoder** | Self-Attention ($Q, K, V$) | $12 \text{ layers} \times 3 \times (768^2 + 768)$ | $21.26$ Million |
| | Attention Output Dense | $12 \text{ layers} \times (768^2 + 768)$ | $7.09$ Million |
| | Attention LayerNorm | $12 \text{ layers} \times (2 \times 768)$ | $18,432$ |
| | Feed-Forward Layer 1 | $12 \text{ layers} \times (768 \cdot 3072 + 3072)$ | $28.35$ Million |
| | Feed-Forward Layer 2 | $12 \text{ layers} \times (3072 \cdot 768 + 768)$ | $28.32$ Million |
| | Feed-Forward LayerNorm | $12 \text{ layers} \times (2 \times 768)$ | $18,432$ |
| **3. Situation MLP** | Compression Dense 1 | $768 \times 512 + 512$ | $393,728$ |
| | Intermediate LayerNorm | $2 \times 512$ | $1,024$ |
| | Refinement Dense 2 | $512 \times 512 + 512$ | $262,656$ |
| | Final Situation LayerNorm | $2 \times 512$ | $1,024$ |
| **4. Option Converter** | Shared Encoder Weights | Re-uses Component 1 & 2 weights | $0$ (Tied) |
| | Option Projector MLP | $768 \times 512 + 512$ | $393,728$ |
| **5. Final Head** | Bilinear Alignment ($W$) | $512 \times 512$ matrix | $262,144$ |
| | Learnable Temperature ($\tau$) | $1 \text{ scalar}$ | $1$ |
| | Parallel MatMul + Softmax | Deterministic Linear Algebra | $0$ (Pure Math) |
| **TOTAL** | | | $\mathbf{\approx 110.2 \text{ Million}}$ |

---

## ⚙️ Detailed Mathematical Walkthrough

### 1. Input Embeddings ($\approx 23.8 \text{M Parameters}$)
The goal is to convert raw token IDs into dense, position-aware vectors.

*   **Token Embeddings:** A lookup table mapping vocabulary tokens to 768-dimensional vectors.
    $$\text{Parameters} = 30,522 \times 768 = 23,440,896$$
*   **Position Embeddings:** Provides sequential awareness across the context window.
    $$\text{Parameters} = 512 \times 768 = 393,216$$
*   **Segment Embeddings & LayerNorm:** Distinguishes context/query segments and applies normalization.
    $$\text{Parameters} = (2 \times 768) + (2 \times 768) = 3,072$$

### 2. The 12-Layer Transformer Backbone ($\approx 85.0 \text{M Parameters}$)
This is the core language understanding engine, utilizing self-attention and feed-forward networks.

*   **Self-Attention Sub-Layer:** For each layer, this computes Query ($Q$), Key ($K$), and Value ($V$) projections.
    $$\text{Parameters per layer} = 3 \times (768^2 + 768) = 1,771,776$$
*   **Attention Output Dense:** Combines the attention outputs across all heads.
    $$\text{Parameters per layer} = (768^2 + 768) = 590,592$$
*   **LayerNorm:** Stabilizes activations within the layer.
    $$\text{Parameters per layer} = 2 \times 768 = 1,536$$
*   **Feed-Forward Network (FFN):** This expands and then contracts the feature representation.
    *   **Layer 1 (Expansion):** $768 \rightarrow 3,072$
        $$\text{Parameters per layer} = (768 \times 3,072 + 3,072) = 2,362,368$$
    *   **Layer 2 (Projection):** $3,072 \rightarrow 768$
        $$\text{Parameters per layer} = (3,072 \times 768 + 768) = 2,360,064$$

**Total Calculation for 12 Layers:**
$$\text{Total per layer} = 1,771,776 (\text{Attention}) + 590,592 (\text{Output}) + 1,536 (\text{Norm}) + 2,362,368 (\text{FFN1}) + 2,360,064 (\text{FFN2}) + 1,536 (\text{Norm}) \approx 7,087,872$$
$$\mathbf{\text{Total for 12 layers}} = 12 \times 7,087,872 \approx \mathbf{85.05 \text{ Million}}$$

### 3. Situation MLP ($\approx 0.66 \text{M Parameters}$)
This network is responsible for compressing the rich text representation into a concise, fixed-size state vector ($z$).

*   **Compression Dense 1:**
    $$\text{Parameters} = (768 \times 512) + 512 = 393,728$$
*   **Intermediate LayerNorm:**
    $$\text{Parameters} = 2 \times 512 = 1,024$$
*   **Refinement Dense 2:**
    $$\text{Parameters} = (512 \times 512) + 512 = 262,656$$
*   **Final LayerNorm:**
    $$\text{Parameters} = 2 \times 512 = 1,024$$

### 4. Option Matrix Converter ($\approx 0.39 \text{M Parameters}$)
This component ensures that candidate options are projected into the same vector space as the situation vector by re-using the core components, thereby minimizing distributional mismatch.

*   **Option Projector MLP:**
    $$\text{Parameters} = (768 \times 512) + 512 = 393,728$$

### 5. Final Head ($\approx 0.26 \text{M Parameters}$)
This final stage computes the compatibility score between the state vector and the options for final decision scoring.

*   **Bilinear Alignment Matrix ($W$):** This matrix captures the feature trade-offs between the situation and options.
    $$\text{Parameters} = 512 \times 512 = 262,144$$
*   **Learnable Temperature ($\tau$):** A single scalar that scales the final probability distribution.
    $$\text{Parameters} = 1$$
*   **Parallel Matrix Multiplication:** The core computation involves a matrix product followed by a Softmax function (which is pure mathematical operation without trainable weights).
    $$\text{Parameters} = 0$$

---

## 💾 Memory Footprint Analysis

The model's memory usage depends heavily on the precision used for deployment.

*   **FP32 Precision (4 bytes/parameter):** $110,204,593 \times 4 \approx \mathbf{440.8 \text{ MB}}$
*   **FP16 / BF16 Precision (2 bytes/parameter):** $110,204,593 \times 2 \approx \mathbf{220.4 \text{ MB}}$
*   **INT8 Quantization (1 byte/parameter):** $110,204,593 \times 1 \approx \mathbf{110.2 \text{ MB}}$

**Conclusion:** This profile is optimized for efficient deployment, fitting comfortably within modest consumer hardware or embedded edge systems, leaving ample VRAM for batch processing.


