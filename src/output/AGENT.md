# AGENT.md — `src/output/`

## Purpose
Takes the compressed **situation vector** `z ∈ ℝ^512` and a list of **candidate action strings**, converts everything to comparable unit vectors, scores them, and returns a fully typed decision result.

---

## Files

### `optionMatrixConvertor.py`

**Class: `OptionMatrixConvertor`**
Converts a list of action strings into a normalized matrix `M ∈ ℝ^{K×512}` where each row is a candidate projected onto the unit hypersphere S^511.

**Constructor:**
```python
from src.output.optionMatrixConvertor import OptionMatrixConvertor

convertor = OptionMatrixConvertor(
    tokenizer=tok,          # BPETokenizer instance
    embedder=embedder,      # EmbeddingTable instance
    hidden_dim=768,         # OPTION_EMBED_DIM
    out_dim=512,            # OPTION_OUT_DIM
    seed=42,                # OPTION_PROJ_SEED — deterministic W_proj init
    eps=1e-12               # OPTION_NORM_EPS
)
```

**Key method:**
```python
M = convertor.convert(["Go left", "Go right", "Stop"])
# M.shape == (3, 512), each row has unit L2 norm
```

**Internal pipeline per call:**
```
action strings
  → BPETokenizer.encode()          token IDs per string
  → EmbeddingTable.forward()       [K, T, 768] embeddings
  → masked mean pooling            [K, 768]    ← stored as self._last_pooled
  → W_proj @ + b_proj              [K, 512]    ← stored as self._last_projected
  → L2 normalize                   [K, 512]    ← stored as self._last_norms
```

**Backprop cache** (used by `DPOTrainer`):
| Attribute | Shape | Contains |
|---|---|---|
| `self._last_pooled` | `[K, 768]` | Mean-pooled embeddings before projection |
| `self._last_projected` | `[K, 512]` | Pre-normalization projected vectors |
| `self._last_norms` | `[K, 1]` | L2 norms of projected vectors |

**Trainable weights:** `W_proj [768, 512]`, `b_proj [512]`

**Alias:** `OptionMatrixConverter = OptionMatrixConvertor` (backwards compat)

---

### `finalHeadAndOutput.py`

**Class: `FinalHeadAndOutput`**
Computes the final ranked decision from a situation vector and options matrix.

**Constructor:**
```python
from src.output.finalHeadAndOutput import FinalHeadAndOutput

head = FinalHeadAndOutput(
    temperature=0.07,       # DEFAULT_TEMPERATURE
    eps=1e-12,              # SIMILARITY_EPS
    num_simulations=64,     # MCTS_NUM_SIMULATIONS (only used if enable_thinking=True)
    c_puct=1.414            # MCTS_C_PUCT
)
```

**Key method:**
```python
result: FinalDecisionOutputDTO = head.forward(
    situation_vector=z,                # shape [1, 512]
    options_matrix=M,                  # shape [K, 512]
    candidate_strings=["opt A", ...],  # list of strings OR List[CandidateOptionDTO] OR OptionMatrixRequestDTO
    enable_thinking=False,             # True → uses MCTS for deeper search
    query_id=None
)
```

**What it computes:**
1. L2-normalizes `z` and `M`
2. Calls `RLPolicy.evaluate()` → softmax probabilities `[1, K]` + logits `[1, K]`
3. Computes cosine similarities `[1, K]`
4. Computes Shannon entropy and confidence margin
5. Returns `FinalDecisionOutputDTO` with `selected_action`, `rankings`, `metrics`, `distribution`

**Dependency:** `src/transformerAndRL/RLPolicy.py` (`RLPolicy`)

---

### `__init__.py`
Exports: `FinalHeadAndOutput`, `OptionMatrixConvertor`, `OptionMatrixConverter`, and all output DTOs.
