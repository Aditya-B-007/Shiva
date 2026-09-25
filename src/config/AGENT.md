# AGENT.md — `src/config/`

## Purpose
Single source of truth for **all constants, typed configs, and data-transfer objects**. Every other module imports from here — never hardcode numbers elsewhere.

---

## Files

### `config.py`
Architectural constants and structured dataclass configurations.

**Top-level constants:**

| Constant | Value | Used for |
|---|---|---|
| `HIDDEN_DIM` | `768` | Transformer embedding width |
| `NUM_LAYERS` | `12` | Encoder layer count |
| `NUM_HEADS` | `12` | Attention heads |
| `HEAD_DIM` | `64` | Per-head width (768 // 12) |
| `FFN_DIM` | `3072` | SwiGLU inner dimension (4 × 768) |
| `SITUATION_DIM` | `512` | Compressed decision vector |
| `ROPE_THETA` | `10000.0` | RoPE base frequency |
| `LAYER_NORM_EPS` | `1e-12` | LayerNorm stability epsilon |
| `SILU_CLIP_BOUND` | `30.0` | Numerical clip for SiLU |
| `PAD_TOKEN` | `"<pad>"` | Padding token string |
| `UNK_TOKEN` | `"<unk>"` | Unknown token string |
| `BOS_TOKEN` | `"<s>"` | Begin-of-sequence token |
| `EOS_TOKEN` | `"</s>"` | End-of-sequence token |
| `DEFAULT_VOCAB_SIZE` | `30522` | BPE target size (spec) |
| `MIN_MERGE_FREQUENCY` | `2` | Minimum pair frequency for BPE merge |
| `OPTION_EMBED_DIM` | `768` | Option embedding input width |
| `OPTION_OUT_DIM` | `512` | Option projected output width |
| `OPTION_PROJ_SEED` | `42` | Option projection matrix seed |
| `DEFAULT_TEMPERATURE` | `0.07` | Softmax temperature |
| `MCTS_NUM_SIMULATIONS` | `64` | MCTS rollout count |
| `MCTS_C_PUCT` | `1.414` | PUCT exploration constant |

**Dataclass configs** (all have `.to_dict()`):
- `TransformerConfig` — groups transformer architectural constants
- `TokenizerConfig` — groups tokenizer constants
- `RAGConfig` — chunking and retrieval hypers
- `OptionConverterConfig` — option projection settings
- `FinalHeadConfig` — temperature, eps, min/max options
- `RLPolicyConfig` — MCTS simulation settings
- `ModelConfig` — master container with all of the above as fields

**Global singleton:**
```python
from src.config.config import default_config
print(default_config.transformer.hidden_dim)  # 768
```

---

### `dtos.py`
All **Pydantic v2 BaseModel** data-transfer objects. Import these for type-safe data passing.

| Class | Purpose |
|---|---|
| `UserPromptDTO` | Wraps a user instruction string |
| `DataUploadDTO` | PDF content + filename + optional prompt |
| `CandidateOptionDTO` | One candidate action: `id`, `text`, `category` |
| `OptionMatrixRequestDTO` | List of 2–32 `CandidateOptionDTO` + `query_id` |
| `RetrievedChunkDTO` | One RAG chunk: `chunk_id`, `score`, `cosine_sim`, `phrase_overlap`, `text` |
| `RetrievalResponseDTO` | Query + list of `RetrievedChunkDTO` |
| `OptionScoreDTO` | Per-option score: rank, probability, logit, cosine_sim |
| `SelectedActionDTO` | Winning action: confidence, margin_over_runner_up |
| `DecisionMetricsDTO` | K, temperature, top prob, entropy, margin |
| `DecisionDistributionDTO` | Full `[1, K]` probability + logit matrices |
| `FinalDecisionOutputDTO` | Root output: selected + rankings + metrics + distribution |
| `DecisionResultDTO` | Alias for `FinalDecisionOutputDTO` (backwards compat) |
| `ScoredActionDTO` | Internal RL policy scored action |
| `DecisionOutputDTO` | Raw output from `RLPolicy.evaluate()` |

---

### `__init__.py`
Re-exports everything from both files. Prefer importing from `src.config` directly:
```python
from src.config import FinalDecisionOutputDTO, HIDDEN_DIM, default_config
```
