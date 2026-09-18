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
│   ├── data.txt                      # 9.1 MB domain text corpus (architecture, code, robotics)
│   └── nandiTrain.jsonl              # 500 curated Q&A pairs with thoughts & human-readable responses
│
├── model_artifacts/
│   ├── tokeniser/                    # Serialized BPE tokenizer model (tokeniser.json)
│
├── src/
│   ├── tokenization.py               # Custom Byte-Level BPE Tokenizer (TokenizerNandi)
│   ├── dataIngestionPipeline.py      # Chunked PyTorch Dataset & DataLoader pipeline
│   ├── transformer.py                # 30.2M RoPE + FlashAttention Causal Transformer architecture
│   └── chat.py                       # Interactive terminal chat & completion interface
│
├── training/
│   ├── trainingNandiOnData.py        # Stage 1: Pre-training on domain corpus with MPS acceleration
│   └── finetuningNandi.py            # Stage 2: Supervised Fine-Tuning (SFT) on Q&A reasoning pairs
│
└── README.md
```

---
All the files are not listed and present citing the size of the files such as the model weights.
--

## ⚙️ Model Specifications (Lightweight SLM)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `ntoken` | `9,714` (custom domain BPE) | Vocabulary size |
| `ninp` (`d_model`) | `512` | Hidden representation dimension |
| `nhead` | `8` | Attention heads (`head_dim = 64`) |
| `nhid` | `2048` | MLP hidden feed-forward dimension |
| `nlayers` | `8` | Transformer blocks (optimized for thermals & throughput) |
| `dropout` | `0.1` | Dropout rate |
| `max_seq_len` | `8192` | RoPE precomputed cache capacity |
| **Embedding Table** | **4.97M params** | Only 16.5% of model (tied `decoder.weight = encoder.weight`) |
| **Transformer Core**| **25.18M params** | 83.5% of capacity dedicated to causal attention & reasoning |
| **Total Parameters** | **30,156,800 (~30.2M)** | Fast & cool training on Apple Silicon, Intel, or NVIDIA GPUs |
| **Batch Convention** | `(batch_size, seq_len)` | Standard `batch_first=True` |

---

<img width="1122" height="665" alt="Screenshot 2026-09-07 at 11 50 40 PM" src="https://github.com/user-attachments/assets/20481296-24dc-4012-bbca-d808beeb103d" />


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

### 5. Stage 1: Base Language Pre-Training

Train the base causal language model on domain text (`data/data.txt`):

```bash
python3 training/trainingNandiOnData.py
```

Checkpoints will be saved automatically to `model_artifacts/checkpoints/nandi_final.pt`.

### 6. Stage 2: Supervised Fine-Tuning (SFT) on Q&A / Reasoning Data

Fine-tune the pre-trained weights on the curated question-and-answer pairs with `<|thought|>` reasoning traces:

```bash
python3 training/finetuningNandi.py
```

Checkpoints will be saved to `model_artifacts/checkpoints/nandi_chat_final.pt`.

### 7. Stage 3: Vision-Language Alignment & Multimodal Training

Train the SigLIP vision bridge and multimodal reasoning capabilities on images:

```bash
# Stage 3A: Alignment warmup (SLM backbone frozen, train MLP Projector only)
python3 training/stage1AlignmentNandi.py

# Stage 3B: End-to-end multimodal fine-tuning (Joint SLM + Projector training)
python3 training/stage2MultimodalNandi.py
```

Checkpoints:
- Stage 1 outputs to `model_artifacts/checkpoints/nandi_stage1_projector.pt`
- Stage 2 outputs final weights to `model_artifacts/checkpoints/nandi_vision_final.pt`

### 8. End-to-End Testing & Verification

Run the full end-to-end verification suite covering text generation, SigLIP vision encoding, multimodal splicing, and optimization convergence:

```bash
python3 tests/test_end_to_end_pipeline.py
```

### 9. Interactive Chat & Vision UI

Launch the ChatGPT-style interface with pure image recognition & chat:

```bash
python3 src/chat.py
```

---

## 🧠 Reasoning & CoT (Chain of Thought)

Nandi incorporates `<|thought|>` delimiters within its tokenization vocabulary and data pipelines to support multi-step reasoning traces prior to final solution dispatch.

