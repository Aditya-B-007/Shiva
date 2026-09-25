# AGENT.md — `training/`

## Purpose
Everything needed to **train and checkpoint the model**. Two training stages (DPO then RLCD) plus the optimizer and one-click pipeline script. Nothing here is used at inference time — import from `src/` for that.

---

## Essential Commands

```bash
# Run the full end-to-end training pipeline (from project root)
python3 training/trainingScript.py

# Syntax-check all training files
python3 -m py_compile training/optimizer.py training/DPO.py training/RLCD.py training/trainingScript.py
```

---

## Files

### `optimizer.py`

**Class: `AdamW`**
Standard AdamW optimizer with gradient clipping. Pure NumPy.

**Constructor:**
```python
from training.optimizer import AdamW

opt = AdamW(
    params=[("W_gate", mlp.W_gate), ("b_gate", mlp.b_gate), ...],
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    max_grad_norm=1.0      # clips gradient norm before update
)
```

**Usage:** `opt.step(grads_dict)` — `grads_dict` maps param name → gradient array. Params are updated **in-place** via `param -=`.

---

### `DPO.py`

**Stage 1 — Direct Preference Optimization**

**Dataclass: `DPOSample`**
```python
@dataclass
class DPOSample:
    context_vector: np.ndarray   # [1, 768] — encoded scenario
    action_strings: List[str]    # candidate option texts
    winner_idx: int              # index of preferred action
    loser_idx: int               # index of rejected action
    ref_logits: np.ndarray       # [1, K] — frozen reference distribution
```

**Class: `DPOTrainer`**

Loss: `L = -log σ(β · (π_margin − ref_margin))`

| Method | What it does |
|---|---|
| `train_step(sample) → (loss, margin)` | Forward pass, analytical backward, AdamW step |
| `_backward_final_mlp(x, grad_z)` | Analytically computes gradients for W_gate, W_up, W_down |

**Trains:** `FinalMLP` weights (`W_gate`, `W_up`, `W_down`) + `OptionMatrixConvertor` weights (`W_proj`, `b_proj`)

**Backprop reads:** `convertor._last_pooled` (set during `convert()`) for `grad_W_proj = _last_pooled.T @ proj_grad`

---

### `RLCD.py`

**Stage 2 — Reinforcement Learning Calibration Distillation**

**Dataclass: `RLCDSample`**
```python
@dataclass
class RLCDSample:
    z_positive: np.ndarray     # [1, 512] — preferred option embedding (teacher)
    z_negative: np.ndarray     # [1, 512] — raw MLP output (student to calibrate)
    options_matrix: np.ndarray # [K, 512] — full candidate matrix
```

**Class: `RLCDTrainer`**

Loss: cross-entropy distillation — student (calibrated `z_neg`) learns to match teacher (preferred `z_pos`) distribution over options.

Learns a linear calibration head: `z_cal = z_neg @ W_cal + b_cal`

`W_cal` initializes as identity `[512, 512]` — starts as a no-op, learns a correction.

| Method | What it does |
|---|---|
| `train_step(sample) → loss` | Forward + analytical grad + AdamW step |

---

### `trainingScript.py`

**Class: `TrainingPipeline`**
Orchestrates both training stages and checkpoint I/O.

| Method | Signature | What it does |
|---|---|---|
| `run_stage1_dpo(dataset, epochs)` | `List[DPOSample], int` | Runs DPO loop, prints epoch losses |
| `run_stage2_rlcd(dataset, epochs)` | `List[RLCDSample], int` | Runs RLCD loop, prints epoch losses |
| `save_checkpoint(filepath, encoder, embedder)` | `str, optional, optional` | Saves all weights to `.npz` (MLP + convertor + RLCD + encoder + embedder) |
| `load_checkpoint(filepath, encoder, embedder)` | `str, optional, optional` | Restores all weights from `.npz` |
| `predict(scenario, options, tokenizer, embedder, encoder)` | → `FinalDecisionOutputDTO` | End-to-end inference |

**Helper functions:**
- `encode_text_to_context(text, tok, embedder, encoder)` → `[1, 768]` — full encode pipeline in one call
- `load_jsonl_dataset(filepath)` → `List[Dict]` — reads + validates `train_dataset.jsonl`
- `build_dpo_samples(records, ...)` → `List[DPOSample]` — constructs DPO training set
- `build_rlcd_samples(records, ...)` → `List[RLCDSample]` — constructs RLCD training set

**`main()` pipeline (what happens when you run the script):**
1. `np.random.seed(42)` — determinism
2. Load tokenizer from JSON if exists, else train from `data.txt`
3. Build `EmbeddingTable`, `BidirectionalEncoderStack`, `FinalMLP`, `OptionMatrixConvertor` with `seed=42`
4. Load JSONL → build `DPOSample` list → run Stage 1 DPO (10 epochs)
5. Build `RLCDSample` list → run Stage 2 RLCD (3 epochs)
6. Save full checkpoint to `model_artifacts/checkpoints/sankalpa_weights.npz`
7. Run one inference sanity-check on `records[0]` and print result

**Config variables at top of script (edit to tune training):**

| Variable | Default | Effect |
|---|---|---|
| `TARGET_VOCAB_SIZE` | `500` | BPE vocabulary target |
| `DPO_EPOCHS` | `10` | Stage 1 epochs |
| `RLCD_EPOCHS` | `3` | Stage 2 epochs |
| `DPO_LR` | `2e-4` | Stage 1 learning rate |
| `RLCD_LR` | `5e-4` | Stage 2 learning rate |
| `TEMPERATURE` | `0.07` | Softmax temperature |
