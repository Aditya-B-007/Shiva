# AGENT.md — `model_artifacts/`

## Purpose
Persisted outputs from training. These files let the model restart without retraining from scratch.

---

## Sub-folders

### `tokenizer/`

#### `tokenizer.json`
Saved BPE vocabulary. Written by `BPETokenizer.save()`, read by `BPETokenizer.load_from_json()`.

**Structure:**
```json
{
  "special_tokens": ["<pad>", "<unk>", "<s>", "</s>"],
  "vocab_size": 433,
  "merges": [
    [["65", "20"], "6520"],   // each entry: [[hex_a, hex_b], hex_merged]
    ...                        // 173 merge rules
  ],
  "token_to_id": {
    "hex_bytes": int_id,       // 433 entries total
    ...
  }
}
```

**Loading in Python:**
```python
from src.transformerAndRL.tokenizer import BPETokenizer

# Option A — classmethod constructor
tok = BPETokenizer.from_json("model_artifacts/tokenizer/tokenizer.json")

# Option B — load into existing instance
tok = BPETokenizer()
tok.load_from_json("model_artifacts/tokenizer/tokenizer.json")

print(tok.vocab_size)   # 433
print(tok.encode("hello world"))
```

> [!WARNING]
> If you delete this file, the next training run will retrain the tokenizer from `data/data.txt`.
> This will produce a **different vocabulary** unless the corpus is identical, which will break
> compatibility with existing embedding weights.

---

### `checkpoints/`

#### `sankalpa_weights.npz`
All learned weight tensors saved by `TrainingPipeline.save_checkpoint()`.

**Keys in the archive:**

| Key | Shape | Component |
|---|---|---|
| `mlp_W_gate` | `[768, 512]` | FinalMLP gate projection |
| `mlp_b_gate` | `[512]` | FinalMLP gate bias |
| `mlp_W_up` | `[768, 512]` | FinalMLP up projection |
| `mlp_b_up` | `[512]` | FinalMLP up bias |
| `mlp_W_down` | `[512, 512]` | FinalMLP down projection |
| `mlp_b_down` | `[512]` | FinalMLP down bias |
| `opt_W_proj` | `[768, 512]` | OptionMatrixConvertor projection |
| `opt_b_proj` | `[512]` | OptionMatrixConvertor bias |
| `cal_W` | `[512, 512]` | RLCD calibration matrix |
| `cal_b` | `[512]` | RLCD calibration bias |
| `enc_l{0-11}_W_q` | `[768, 768]` | Per-layer Q projection (×12) |
| `enc_l{0-11}_W_k` | `[768, 768]` | Per-layer K projection (×12) |
| `enc_l{0-11}_W_v` | `[768, 768]` | Per-layer V projection (×12) |
| `enc_l{0-11}_W_o` | `[768, 768]` | Per-layer output projection (×12) |
| `enc_l{0-11}_W_gate/up/down` | various | Per-layer FFN weights (×12) |
| `enc_l{0-11}_norm1/2_gamma/beta` | `[768]` | Per-layer LayerNorm (×12) |
| `embedding_weights` | `[433, 768]` | EmbeddingTable lookup matrix |

**Inspecting keys:**
```bash
python3 -c "
import zipfile
z = zipfile.ZipFile('model_artifacts/checkpoints/sankalpa_weights.npz')
for name in z.namelist():
    print(name, z.getinfo(name).file_size, 'bytes')
"
```

**Loading in Python:**
```python
import numpy as np
data = np.load("model_artifacts/checkpoints/sankalpa_weights.npz")
print(data["mlp_W_gate"].shape)   # (768, 512)
print(list(data.keys())[:5])
```
