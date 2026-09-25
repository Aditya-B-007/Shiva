# AGENT.md — Project Root

## What Is This?
**Sankalpa (110M)** — a pure-NumPy, zero-framework transformer that reads a situation and picks the best action from a candidate list. No PyTorch, no TensorFlow. Every matrix multiply is explicit.

---

## 30-Second Pipeline

```
Raw Text (data.txt)
    ↓  BPETokenizer.train_from_file()
Vocabulary (tokenizer.json, 433 pieces)
    ↓  EmbeddingTable.forward()
Token Embeddings [B, T, 768]
    ↓  BidirectionalEncoderStack.forward()   ← 12 layers, RoPE, SwiGLU
Hidden States [B, T, 768]
    ↓  mean_pooling() → FinalMLP.forward()
Situation Vector z ∈ ℝ^512
    ↓  DPO (Stage 1) + RLCD (Stage 2) training
Trained z
    ↓  OptionMatrixConvertor.convert()
Options Matrix M ∈ ℝ^{K×512}  (each row is one candidate on unit sphere S^511)
    ↓  FinalHeadAndOutput.forward()
FinalDecisionOutputDTO   ← winner + full ranked list + confidence metrics
```

---

## Folder Map

| Folder | Role |
|---|---|
| `data/` | Raw training corpus + JSONL preference dataset |
| `model_artifacts/` | Saved tokenizer vocab + all weight checkpoints |
| `src/config/` | All constants, dataclass configs, Pydantic DTOs |
| `src/dataUpload/` | PDF ingestion + hybrid TF-IDF RAG retriever |
| `src/output/` | Option embedding → decision scoring head |
| `src/transformerAndRL/` | BPE tokenizer, 12-layer encoder, MCTS RL policy |
| `training/` | AdamW optimizer, DPO trainer, RLCD trainer, one-click script |

---

## Essential Commands

```bash
# Run full end-to-end training (loads tokenizer if it exists, trains otherwise)
python3 training/trainingScript.py

# Syntax-check all source files
python3 -m py_compile src/transformerAndRL/tokenizer.py \
    src/transformerAndRL/transformer.py \
    src/output/optionMatrixConvertor.py \
    training/trainingScript.py

# Inspect what keys are in the checkpoint
python3 -c "
import zipfile
z = zipfile.ZipFile('model_artifacts/checkpoints/sankalpa_weights.npz')
print(z.namelist())
"

# Verify tokenizer vocab size
python3 -c "
import json
d = json.load(open('model_artifacts/tokenizer/tokenizer.json'))
print('vocab_size:', d['vocab_size'], '| merges:', len(d['merges']))
"

# Check project file tree
find . -not -path '*/__pycache__/*' -not -path '*/.git/*' -not -name '*.pyc' | sort
```

---

## Key Design Constants (all in `src/config/config.py`)

| Constant | Value | Meaning |
|---|---|---|
| `HIDDEN_DIM` | 768 | Transformer model width |
| `NUM_LAYERS` | 12 | Encoder depth |
| `NUM_HEADS` | 12 | Attention heads |
| `SITUATION_DIM` | 512 | Compressed decision vector size |
| `DEFAULT_TEMPERATURE` | 0.07 | Softmax sharpness |
| `SEED` (training) | 42 | Fixed for determinism |

---

## Cross-Folder Dependencies

```
training/trainingScript.py
  imports → src/transformerAndRL/{tokenizer, transformer}
  imports → src/output/{optionMatrixConvertor, finalHeadAndOutput}
  imports → training/{DPO, RLCD, optimizer}
  reads   → data/
  writes  → model_artifacts/
```
