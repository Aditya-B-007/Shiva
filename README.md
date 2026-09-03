**Nandi**
The Small Language Model that allows the user to interact and aid in describing to the SLM and connecting shiva.ai with the control problem easily.

Folder structure:

nandi/
│
├── data/
│   └── business_dataset.txt         # Your raw English enterprise text files
│
├── model_artifacts/
│   └── tokenizer/                   # Generated automatically after step 1
│       ├── vocab.json               # Word-to-ID lookup dictionary (50,257 rows)
│       └── merges.txt               # BPE rules file
│
├── src/
│   ├── __init__.py
│   │
│   ├── tokenizer.py                 # Script 1: Trains and saves tokenizer artifacts
│   ├── dataset.py                   # Script 2: Custom PyTorch Dataset/DataLoader (truncates to 1024)
│   │
│   ├── model/                       # Your Neural Network Core
│   │   ├── __init__.py
│   │   ├── embeddings.py            # Layer 1: Token & Positional embeddings (896 dim + Weight Tying)
│   │   ├── attention.py             # Layer 2: Causal Multi-Head Attention (14 heads)
│   │   ├── ffn.py                   # Layer 3: Feed-Forward Network (3,584 dim with GELU)
│   │   ├── block.py                 # Layer 4: Standard Transformer Block (combines attention + ffn + LayerNorm)
│   │   └── lm_head.py               # Layer 5: Final output layer mapping thoughts back to 50,257 tokens
│   │
│   └── config.py                    # Holds hyperparameter constants (d_model=896, heads=14, etc.)
│
├── train.py                         # Master training execution script (runs the loops)
└── requirements.txt                 # torch, tokenizers
