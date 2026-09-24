import numpy as np
from typing import List, Union, Optional
try:
    from src.config.config import (
        OPTION_EMBED_DIM,
        OPTION_OUT_DIM,
        OPTION_PROJ_SEED,
        OPTION_NORM_EPS,
        OptionConverterConfig,
    )
except ImportError:
    from Shiva.src.config.config import (
        OPTION_EMBED_DIM,
        OPTION_OUT_DIM,
        OPTION_PROJ_SEED,
        OPTION_NORM_EPS,
        OptionConverterConfig,
    )

class OptionMatrixConvertor:
    def __init__(
        self,
        tokenizer,
        embedder,
        hidden_dim: int = OPTION_EMBED_DIM,
        out_dim: int = OPTION_OUT_DIM,
        seed: int = OPTION_PROJ_SEED,
        eps: float = OPTION_NORM_EPS
    ):
        self.tokenizer = tokenizer
        self.embedder = embedder
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.eps = eps

        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / (hidden_dim + out_dim))
        self.W_proj = rng.randn(hidden_dim, out_dim).astype(np.float32) * scale
        self.b_proj = np.zeros(out_dim, dtype=np.float32)

    def convert(self, action_strings: List[str]) -> np.ndarray:
        if hasattr(action_strings, "options"):
            action_strings = [opt.text for opt in action_strings.options]
        elif isinstance(action_strings, list):
            action_strings = [opt.text if hasattr(opt, "text") else str(opt) for opt in action_strings]

        K = len(action_strings)
        if K == 0:
            return np.empty((0, self.out_dim), dtype=np.float32)

        # 1. Tokenize all action strings
        tokenized_actions = [self.tokenizer.encode(action) for action in action_strings]
        
        # Ensure at least length 1 to prevent empty zero-dim arrays
        max_seq_len = max(max(len(toks) for toks in tokenized_actions), 1)

        # 2. Vectorized Batching and Padding (Pad token ID = 0)
        batch_ids = np.zeros((K, max_seq_len), dtype=np.int32)
        attention_mask = np.zeros((K, max_seq_len), dtype=np.float32)

        for i, token_ids in enumerate(tokenized_actions):
            if len(token_ids) > 0:
                batch_ids[i, :len(token_ids)] = token_ids
                attention_mask[i, :len(token_ids)] = 1.0
            else:
                batch_ids[i, 0] = 0
                attention_mask[i, 0] = 0.0

        # 3. Shared Embedding Lookup: [K, max_seq_len, 768]
        embeddings = self.embedder.forward(batch_ids)

        # 4. Masked Mean Pooling: [K, 768]
        mask_expanded = attention_mask[:, :, np.newaxis]  # [K, max_seq_len, 1]
        sum_embeddings = np.sum(embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
        pooled_vectors = sum_embeddings / sum_mask  # [K, 768]
        self._last_pooled = pooled_vectors

        # 5. Linear Projection: [K, 768] @ [768, 512] -> [K, 512]
        projected = np.matmul(pooled_vectors, self.W_proj) + self.b_proj
        self._last_projected = projected

        # 6. Radial L2 Normalization onto the unit sphere S^511
        norms = np.linalg.norm(projected, ord=2, axis=-1, keepdims=True) + self.eps
        self._last_norms = norms
        options_matrix = projected / norms

        return options_matrix.astype(np.float32)


OptionMatrixConverter = OptionMatrixConvertor
__all__ = ["OptionMatrixConvertor", "OptionMatrixConverter"]