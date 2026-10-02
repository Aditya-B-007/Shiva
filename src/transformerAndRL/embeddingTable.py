import numpy as np
from typing import Optional, Dict, Any
try:
    from src.config.config import HIDDEN_DIM, PAD_TOKEN_ID
except ImportError:
    from Shiva.src.config.config import HIDDEN_DIM, PAD_TOKEN_ID

__all__ = ["EmbeddingTable"]


class EmbeddingTable:

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = HIDDEN_DIM,
        pad_token_id: Optional[int] = PAD_TOKEN_ID,
        seed: Optional[int] = None
    ):
        self._vocab_size: int = vocab_size
        self._hidden_dim: int = hidden_dim
        self._pad_token_id: Optional[int] = pad_token_id

        scale = 1.0 / np.sqrt(hidden_dim)
        rng = np.random.RandomState(seed) if seed is not None else np.random
        self._weights: np.ndarray = rng.normal(
            loc=0.0, scale=scale, size=(vocab_size, hidden_dim)
        ).astype(np.float32)

        if pad_token_id is not None and 0 <= pad_token_id < vocab_size:
            self._weights[pad_token_id] = 0.0

        self._grad_weights: np.ndarray = np.zeros_like(self._weights)
        self._last_input_ids: Optional[np.ndarray] = None

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: Any,
        hidden_dim: int = HIDDEN_DIM,
        seed: Optional[int] = None
    ) -> "EmbeddingTable":
        """Instantiates an EmbeddingTable matching the given tokenizer's vocabulary."""
        return cls(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=hidden_dim,
            pad_token_id=tokenizer.pad_token_id,
            seed=seed
        )

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    @property
    def grad_weights(self) -> np.ndarray:
        return self._grad_weights

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._pad_token_id

    # -----------------------------------------------------------------
    # Forward & Backward
    # -----------------------------------------------------------------

    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        self._last_input_ids = input_ids
        return self._weights[input_ids]

    def backward(self, grad_output: np.ndarray) -> None:
        if self._last_input_ids is None:
            raise RuntimeError("backward() called before forward() on EmbeddingTable.")

        self._grad_weights.fill(0.0)
        np.add.at(self._grad_weights, self._last_input_ids, grad_output)

        if self._pad_token_id is not None and 0 <= self._pad_token_id < self._vocab_size:
            self._grad_weights[self._pad_token_id] = 0.0

    # -----------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------

    def get_weights(self) -> Dict[str, np.ndarray]:
        return {"embedding_weights": self._weights.copy()}

    def set_weights(self, weights: Dict[str, np.ndarray]) -> None:
        if "embedding_weights" in weights:
            loaded = weights["embedding_weights"].astype(np.float32)
            self._weights = loaded
            self._vocab_size = loaded.shape[0]
            self._hidden_dim = loaded.shape[1]
            self._grad_weights = np.zeros_like(self._weights)
            self._last_input_ids = None
