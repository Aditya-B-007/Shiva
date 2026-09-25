# AGENT.md — `src/`

## Purpose
The **inference-time library**. All model architecture, tokenization, data ingestion, and output logic lives here. Nothing in `src/` writes to disk — that is the job of `training/`.

---

## Sub-packages

| Sub-package | What it does |
|---|---|
| `src/config/` | Global constants, typed dataclass configs, all Pydantic DTOs |
| `src/dataUpload/` | PDF text extraction, sliding-window chunking, TF-IDF + phrase RAG |
| `src/output/` | Converts option strings → `[K, 512]` matrix; cosine scoring → `FinalDecisionOutputDTO` |
| `src/transformerAndRL/` | BPE tokenizer, 12-layer bidirectional encoder, MCTS RL policy |

---

## `src/__init__.py`
Empty — does not re-export sub-packages. Import from sub-packages directly:

```python
# Correct
from src.transformerAndRL.tokenizer import BPETokenizer
from src.output.finalHeadAndOutput import FinalHeadAndOutput
from src.config.config import HIDDEN_DIM

# Incorrect — will not work
from src import BPETokenizer
```

---

## Import Path Notes

All files support both direct import (when `Shiva/` is not the root) and package import:

```python
try:
    from src.config.config import HIDDEN_DIM
except ImportError:
    from Shiva.src.config.config import HIDDEN_DIM
```

This pattern appears in every file. It means the code works whether you run from `/Projects/Shiva/` or from `/Projects/`.

---

## Data Flow Through `src/`

```
str input
  → src/transformerAndRL/tokenizer.py   BPETokenizer.encode()
  → src/transformerAndRL/tokenizer.py   EmbeddingTable.forward()      [B, T, 768]
  → src/transformerAndRL/transformer.py BidirectionalEncoderStack      [B, T, 768]
  → src/transformerAndRL/transformer.py mean_pooling + FinalMLP        [1, 512]
  → src/output/optionMatrixConvertor.py OptionMatrixConvertor.convert  [K, 512]
  → src/output/finalHeadAndOutput.py    FinalHeadAndOutput.forward     FinalDecisionOutputDTO
```
