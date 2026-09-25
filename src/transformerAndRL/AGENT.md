# AGENT.md — `src/transformerAndRL/`

## Purpose
The **core neural architecture**: byte-pair encoding tokenizer, learnable embeddings, 12-layer bidirectional transformer encoder, and an MCTS-based RL policy for inference-time reasoning.

---

## Files

### `tokenizer.py`

**Class: `BPETokenizer`**
Learns subword merges from a text corpus; encodes strings to integer token ID sequences.

| Method | Signature | What it does |
|---|---|---|
| `train(corpus, target_vocab_size)` | `List[str], int` | Run BPE merge loop on corpus |
| `train_from_file(file_path, target_vocab_size)` | `str, int` | Read file, call `train()` |
| `encode(text)` | `str → List[int]` | Apply all learned merges, return token IDs |
| `decode(token_ids)` | `List[int] → str` | Reverse lookup, skip special tokens |
| `encode_batch(texts, max_len)` | `List[str] → (ids_array, mask_array)` | Batch encode with padding; returns `[B,T]` arrays |
| `save(filepath)` | `str` | Write vocab + merges to JSON |
| `load(filepath)` | `str` | Load from JSON (in-place) |
| `load_from_json(filepath)` | `str` | Alias for `load()` |
| `from_json(filepath)` | `classmethod → BPETokenizer` | Construct + load in one call |

**Properties:** `vocab_size`, `pad_token_id`

**Special tokens (IDs 0–3):** `<pad>=0`, `<unk>=1`, `<s>=2`, `</s>=3`

---

**Class: `EmbeddingTable`**
Lookup table mapping token IDs → dense float32 vectors.

| Method | Signature | What it does |
|---|---|---|
| `from_tokenizer(tok, hidden_dim, seed)` | `classmethod` | Construct with matching vocab size |
| `forward(input_ids)` | `[B,T] → [B,T,768]` | Index into `self.weights` |
| `backward(grad_output)` | `[B,T,768]` | Accumulate embedding gradients |
| `get_weights()` | `→ Dict` | Export `{"embedding_weights": array}` |
| `set_weights(weights)` | `Dict` | Load from checkpoint dict |

**Constructor params:** `vocab_size`, `hidden_dim=768`, `pad_token_id=0`, `seed=None`

---

### `transformer.py`

**Tensor shapes through the encoder:**
```
Input:   [B, T, 768]  (EmbeddingTable output)
Layer 0–11:           TransformerEncoderLayer
  → MultiHeadAttention  RoPE + L2-normalized QK + softmax → [B, T, 768]
  → LayerNorm + residual
  → FeedForwardNetwork  SwiGLU: silu(x@W_gate) * (x@W_up) @ W_down → [B, T, 768]
  → LayerNorm + residual
Output:  [B, T, 768]
```

**Key classes:**

| Class | Constructor params | Role |
|---|---|---|
| `LayerNormalization` | `hidden_dim, eps` | Standard layer norm with learnable γ, β |
| `MultiHeadAttention` | `hidden_dim, num_heads, seed` | 4 projection matrices + RoPE + cosine QK |
| `FeedForwardNetwork` | `hidden_dim, ffn_dim, seed` | SwiGLU: W_gate, W_up, W_down |
| `TransformerEncoderLayer` | `hidden_dim, num_heads, ffn_dim, seed` | Attention + FFN + 2× LayerNorm |
| `BidirectionalEncoderStack` | `num_layers=12, hidden_dim=768, num_heads=12, ffn_dim=3072, seed` | 12 encoder layers, RoPE precomputed per call |
| `FinalMLP` | `in_features=768, hidden_dim=512, seed` | SwiGLU MLP → LayerNorm; maps `[1,768]→[1,512]` |

**Serialization on `BidirectionalEncoderStack`:**
```python
weights = encoder.get_weights()   # Dict[str, ndarray], 19 keys × 12 layers
encoder.set_weights(weights)      # Restore from dict or np.load() result
```

**Utility functions:**
- `mean_pooling(token_embeddings, attention_mask)` — masked average over T dim
- `softmax(x, axis)` — numerically stable
- `silu(x)` — clipped sigmoid-weighted linear unit
- `precompute_rope_frequencies(head_dim, seq_len, theta)` → `(cos, sin)`

---

### `RLPolicy.py`

**Class: `MCTSSearch`**
Runs PUCT tree search over K candidate actions.
- `search(situation_vector, options_matrix, base_probs)` → `[1, K]` visit-count policy

**Class: `RLPolicy`**
Wraps MCTS + softmax scoring.
- `evaluate(situation_vector, options_matrix, candidate_strings, enable_thinking)` → `DecisionOutputDTO`
- When `enable_thinking=False`: pure softmax (fast)
- When `enable_thinking=True`: MCTS with 64 simulations + Dirichlet noise (thorough)

---

### `__init__.py`
Exports all public classes: `BPETokenizer`, `EmbeddingTable`, `BidirectionalEncoderStack`, `TransformerEncoderLayer`, `MultiHeadAttention`, `FeedForwardNetwork`, `LayerNormalization`, `FinalMLP`, `mean_pooling`, `softmax`, `silu`, `RLPolicy`, `MCTSSearch`, `MCTSNode`
