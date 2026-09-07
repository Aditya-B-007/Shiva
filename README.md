# 🐂 Nandi

**Nandi** is a lightweight, domain-specialized Small Language Model (SLM) designed to bridge the user interface with **Shiva.ai**'s control and orchestration systems. Built with modern transformer innovations, Nandi provides fast, efficient token generation and reasoning for domain-specific control tasks.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/5048a624-fb58-4792-9b83-674dc8014bc4" />

---

## 🏗️ Architectural Overview

Nandi uses a modern **Causal Decoder-Only Transformer** built for speed, stability, and reasoning capabilities:

* **Positional Embeddings**: **RoPE (Rotary Position Embeddings)** applied to queries and keys across all attention heads for superior context and length extrapolation.
* **Attention Kernel**: **PyTorch Scaled Dot-Product Attention (`F.scaled_dot_product_attention`)**, natively leveraging FlashAttention where supported with causal masking.
* **Normalization**: **Pre-Layer Normalization (Pre-LN)** with a final `nn.LayerNorm` prior to the language modeling head for deep network training stability.
* **Activation**: **GELU** (Gaussian Error Linear Unit) inside the Feed-Forward Network.
* **Weight Tying**: Shared weights between input embedding and the output projection matrix (`self.decoder.weight = self.encoder.weight`) to conserve memory and enhance representations.
* **Initialization**: **Orthogonal initialization** across weight matrices with normalized LayerNorm parameters.
* **Reasoning Ready**: Native support for `<|thought|>` delimiter tokens and step-by-step reasoning sequences.

---

## 📂 Project Structure

```text
Shiva/
├── data/
│   ├── data.txt                      # Raw corpus for tokenizer training & ingestion
│   └── nandiTrain.jsonl              # Supervised fine-tuning / reasoning training dataset
│
├── model_artifacts/
│   ├── tokeniser/                    # Serialized BPE tokenizer model (tokeniser.json)
│   └── dataProcessing/               # Data pipeline artifacts & preprocessed caches
│
├── src/
│   ├── tokenization.py               # Custom Byte-Level BPE Tokenizer (TokenizerNandi)
│   ├── dataIngestionPipeline.py      # Chunked PyTorch Dataset & DataLoader pipeline
│   └── transformer.py                # Core Transformer architecture with RoPE & SDPA
│
└── README.md
```

---

## ⚙️ Model Specifications

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `ntoken` | `50,257` (or `8,192` custom BPE) | Vocabulary size |
| `ninp` (`d_model`) | `1024` | Hidden representation dimension |
| `nhead` | `16` | Attention heads (`head_dim = 64`) |
| `nhid` | `4096` | MLP hidden feed-forward dimension |
| `nlayers` | `20` | Transformer blocks |
| `dropout` | `0.1` | Dropout rate |
| `max_seq_len` | `8192` | RoPE precomputed cache capacity |
| **Batch Convention** | `(batch_size, seq_len)` | Standard `batch_first=True` |

---

## 🚀 Quickstart

### 1. Requirements

Ensure you have Python 3.9+ and PyTorch installed:

```bash
pip install torch tokenizers
```

### 2. Tokenizer Training & Testing

The tokenizer uses Byte-Level BPE with custom special tokens (`<unk>`, `<pad>`, `</s>`, `<|thought|>`):

```bash
python3 src/tokenization.py
```

### 3. Data Ingestion Pipeline

To verify the sliding window chunking and batch creation:

```bash
python3 src/dataIngestionPipeline.py
```

### 4. Running the Model Core

Run the model script directly to inspect parameter count and test a forward pass:

```bash
python3 src/transformer.py
```

Example output:
```text
Total Parameters (unique):     218,655,744
Trainable Parameters (unique): 218,655,744
Target Parameter Fit:          218.66 Million Parameters
Output shape successfully verified: torch.Size([4, 32, 50257])
```

---

## 🧠 Reasoning & CoT (Chain of Thought)

Nandi incorporates `<|thought|>` delimiters within its tokenization vocabulary and data pipelines to support multi-step reasoning traces prior to final solution dispatch.

