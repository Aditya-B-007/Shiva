import numpy as np
from typing import Optional, Dict, Any, List
try:
    from src.config.config import (
        HIDDEN_DIM,
        NUM_LAYERS,
        NUM_HEADS,
        HEAD_DIM,
        FFN_DIM,
        SITUATION_DIM,
        ROPE_THETA,
        LAYER_NORM_EPS,
        SILU_CLIP_BOUND,
        TransformerConfig,
    )
except ImportError:
    from Shiva.src.config.config import (
        HIDDEN_DIM,
        NUM_LAYERS,
        NUM_HEADS,
        HEAD_DIM,
        FFN_DIM,
        SITUATION_DIM,
        ROPE_THETA,
        LAYER_NORM_EPS,
        SILU_CLIP_BOUND,
        TransformerConfig,
    )

# =====================================================================
# 1. CORE MATHEMATICAL & ROPE PRIMITIVES
# =====================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -SILU_CLIP_BOUND, SILU_CLIP_BOUND)))


def precompute_rope_frequencies(head_dim: int = HEAD_DIM, seq_len: int = 512, theta: float = ROPE_THETA):
    channel_indices = np.arange(0, head_dim, 2, dtype=np.float32)
    inv_freq = 1.0 / (theta ** (channel_indices / head_dim))  
    
    positions = np.arange(seq_len, dtype=np.float32)          
    freqs = np.outer(positions, inv_freq)                    
    freqs = np.concatenate([freqs, freqs], axis=-1)           
    
    cos = np.cos(freqs)[np.newaxis, np.newaxis, :, :]         
    sin = np.sin(freqs)[np.newaxis, np.newaxis, :, :]         
    return cos.astype(np.float32), sin.astype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
    d_2 = x.shape[-1] // 2
    x1 = x[..., :d_2]
    x2 = x[..., d_2:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    return (x * cos) + (rotate_half(x) * sin)


class LayerNormalization:
    def __init__(self, hidden_dim: int = HIDDEN_DIM, eps: float = LAYER_NORM_EPS):
        self.eps = eps
        self.gamma = np.ones(hidden_dim, dtype=np.float32)
        self.beta = np.zeros(hidden_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta


# =====================================================================
# 2. MULTI-HEAD ATTENTION 
# =====================================================================

class MultiHeadAttention:
    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS, seed: Optional[int] = None):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 768 // 12 = 64
        self.scale = 1.0 / np.sqrt(self.head_dim)

        scale_init = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        rng = np.random.RandomState(seed) if seed is not None else np.random
        
        self.W_q = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale_init
        self.b_q = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W_k = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale_init
        self.b_k = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W_v = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale_init
        self.b_v = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W_o = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale_init
        self.b_o = np.zeros(hidden_dim, dtype=np.float32)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))  # [Batch, Num_Heads, Seq_Len, Head_Dim]

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        x = np.transpose(x, (0, 2, 1, 3))     # [Batch, Seq_Len, Num_Heads, Head_Dim]
        batch_size, seq_len, _, _ = x.shape
        return x.reshape(batch_size, seq_len, self.hidden_dim)

    def forward(
        self, 
        x: np.ndarray, 
        cos: np.ndarray, 
        sin: np.ndarray, 
        attention_mask: np.ndarray = None
    ) -> np.ndarray:
        # Linear projections
        Q = np.matmul(x, self.W_q) + self.b_q
        K = np.matmul(x, self.W_k) + self.b_k
        V = np.matmul(x, self.W_v) + self.b_v

        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        Q_heads = apply_rope(Q_heads, cos, sin)
        K_heads = apply_rope(K_heads, cos, sin)

        eps = 1e-12
        Q_heads = Q_heads / (np.linalg.norm(Q_heads, ord=2, axis=-1, keepdims=True) + eps)
        K_heads = K_heads / (np.linalg.norm(K_heads, ord=2, axis=-1, keepdims=True) + eps)

        scores = np.matmul(Q_heads, np.transpose(K_heads, (0, 1, 3, 2))) * self.scale

        if attention_mask is not None:
            expanded_mask = attention_mask[:, np.newaxis, np.newaxis, :]     
            scores = np.where(expanded_mask == 1, scores, -1e9)
            
        attn_weights = softmax(scores, axis=-1)
        context = np.matmul(attn_weights, V_heads)
        merged_context = self._merge_heads(context)

        return np.matmul(merged_context, self.W_o) + self.b_o


# =====================================================================
# 3. FEED-FORWARD NETWORK (SWIGLU)
# =====================================================================

class FeedForwardNetwork:
    def __init__(self, hidden_dim: int = HIDDEN_DIM, ffn_dim: int = FFN_DIM, seed: Optional[int] = None):
        scale_in = np.sqrt(2.0 / (hidden_dim + ffn_dim))
        scale_out = np.sqrt(2.0 / (ffn_dim + hidden_dim))
        rng = np.random.RandomState(seed) if seed is not None else np.random
        
        self.W_gate = rng.randn(hidden_dim, ffn_dim).astype(np.float32) * scale_in
        self.b_gate = np.zeros(ffn_dim, dtype=np.float32)

        self.W_up = rng.randn(hidden_dim, ffn_dim).astype(np.float32) * scale_in
        self.b_up = np.zeros(ffn_dim, dtype=np.float32)

        self.W_down = rng.randn(ffn_dim, hidden_dim).astype(np.float32) * scale_out
        self.b_down = np.zeros(hidden_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = silu(np.matmul(x, self.W_gate) + self.b_gate)
        up = np.matmul(x, self.W_up) + self.b_up
        return np.matmul(gate * up, self.W_down) + self.b_down


# =====================================================================
# 4. SINGLE ENCODER LAYER
# =====================================================================

class TransformerEncoderLayer:
    def __init__(
        self,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        ffn_dim: int = FFN_DIM,
        seed: Optional[int] = None
    ):
        attn_seed = (seed * 2 + 1) if seed is not None else None
        ffn_seed = (seed * 2 + 2) if seed is not None else None
        self.attention = MultiHeadAttention(hidden_dim, num_heads, seed=attn_seed)
        self.norm1 = LayerNormalization(hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_dim, ffn_dim, seed=ffn_seed)
        self.norm2 = LayerNormalization(hidden_dim)

    def forward(
        self, 
        x: np.ndarray, 
        cos: np.ndarray, 
        sin: np.ndarray, 
        attention_mask: np.ndarray = None
    ) -> np.ndarray:
        attn_out = self.attention.forward(x, cos=cos, sin=sin, attention_mask=attention_mask)
        x = self.norm1.forward(x + attn_out)

        ffn_out = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_out)
        return x


# =====================================================================
# 5. FULL 12-LAYER BIDIRECTIONAL ENCODER STACK
# =====================================================================

class BidirectionalEncoderStack:
    def __init__(
        self,
        num_layers: int = NUM_LAYERS,
        hidden_dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        ffn_dim: int = FFN_DIM,
        seed: Optional[int] = None
    ):
        self.num_layers = num_layers
        self.head_dim = hidden_dim // num_heads
        self.layers = [
            TransformerEncoderLayer(
                hidden_dim,
                num_heads,
                ffn_dim,
                seed=(seed * 100 + i) if seed is not None else None
            )
            for i in range(num_layers)
        ]

    def forward(self, x: np.ndarray, attention_mask: np.ndarray = None) -> np.ndarray:
        seq_len = x.shape[1]
        cos, sin = precompute_rope_frequencies(self.head_dim, seq_len)

        hidden_states = x
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, cos=cos, sin=sin, attention_mask=attention_mask)
        return hidden_states

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Exports weights of all 12 transformer encoder layers."""
        weights: Dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            weights[f"enc_l{i}_W_q"] = layer.attention.W_q.copy()
            weights[f"enc_l{i}_b_q"] = layer.attention.b_q.copy()
            weights[f"enc_l{i}_W_k"] = layer.attention.W_k.copy()
            weights[f"enc_l{i}_b_k"] = layer.attention.b_k.copy()
            weights[f"enc_l{i}_W_v"] = layer.attention.W_v.copy()
            weights[f"enc_l{i}_b_v"] = layer.attention.b_v.copy()
            weights[f"enc_l{i}_W_o"] = layer.attention.W_o.copy()
            weights[f"enc_l{i}_b_o"] = layer.attention.b_o.copy()

            weights[f"enc_l{i}_norm1_gamma"] = layer.norm1.gamma.copy()
            weights[f"enc_l{i}_norm1_beta"] = layer.norm1.beta.copy()

            weights[f"enc_l{i}_W_gate"] = layer.ffn.W_gate.copy()
            weights[f"enc_l{i}_b_gate"] = layer.ffn.b_gate.copy()
            weights[f"enc_l{i}_W_up"] = layer.ffn.W_up.copy()
            weights[f"enc_l{i}_b_up"] = layer.ffn.b_up.copy()
            weights[f"enc_l{i}_W_down"] = layer.ffn.W_down.copy()
            weights[f"enc_l{i}_b_down"] = layer.ffn.b_down.copy()

            weights[f"enc_l{i}_norm2_gamma"] = layer.norm2.gamma.copy()
            weights[f"enc_l{i}_norm2_beta"] = layer.norm2.beta.copy()
        return weights

    def set_weights(self, weights: Dict[str, Any]) -> None:
        """Loads weights for all 12 transformer encoder layers."""
        for i, layer in enumerate(self.layers):
            if f"enc_l{i}_W_q" in weights:
                layer.attention.W_q = weights[f"enc_l{i}_W_q"].astype(np.float32)
                layer.attention.b_q = weights[f"enc_l{i}_b_q"].astype(np.float32)
                layer.attention.W_k = weights[f"enc_l{i}_W_k"].astype(np.float32)
                layer.attention.b_k = weights[f"enc_l{i}_b_k"].astype(np.float32)
                layer.attention.W_v = weights[f"enc_l{i}_W_v"].astype(np.float32)
                layer.attention.b_v = weights[f"enc_l{i}_b_v"].astype(np.float32)
                layer.attention.W_o = weights[f"enc_l{i}_W_o"].astype(np.float32)
                layer.attention.b_o = weights[f"enc_l{i}_b_o"].astype(np.float32)

                layer.norm1.gamma = weights[f"enc_l{i}_norm1_gamma"].astype(np.float32)
                layer.norm1.beta = weights[f"enc_l{i}_norm1_beta"].astype(np.float32)

                layer.ffn.W_gate = weights[f"enc_l{i}_W_gate"].astype(np.float32)
                layer.ffn.b_gate = weights[f"enc_l{i}_b_gate"].astype(np.float32)
                layer.ffn.W_up = weights[f"enc_l{i}_W_up"].astype(np.float32)
                layer.ffn.b_up = weights[f"enc_l{i}_b_up"].astype(np.float32)
                layer.ffn.W_down = weights[f"enc_l{i}_W_down"].astype(np.float32)
                layer.ffn.b_down = weights[f"enc_l{i}_b_down"].astype(np.float32)

                layer.norm2.gamma = weights[f"enc_l{i}_norm2_gamma"].astype(np.float32)
                layer.norm2.beta = weights[f"enc_l{i}_norm2_beta"].astype(np.float32)


# =====================================================================
# 6. POOLING & SITUATION / FINAL MLP 
# =====================================================================

def mean_pooling(token_embeddings: np.ndarray, attention_mask: np.ndarray = None) -> np.ndarray:
    if token_embeddings.ndim == 2:
        return token_embeddings

    if attention_mask is not None:
        expanded_mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * expanded_mask, axis=1)
        sum_mask = np.clip(np.sum(expanded_mask, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
    else:
        return np.mean(token_embeddings, axis=1)


class FinalMLP:
    def __init__(
        self,
        in_features: int = HIDDEN_DIM,
        hidden_dim: int = SITUATION_DIM,
        seed: Optional[int] = None
    ):
        scale_in = np.sqrt(2.0 / (in_features + hidden_dim))
        scale_out = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        rng = np.random.RandomState(seed) if seed is not None else np.random

        self.W_gate = rng.randn(in_features, hidden_dim).astype(np.float32) * scale_in
        self.b_gate = np.zeros(hidden_dim, dtype=np.float32)

        self.W_up = rng.randn(in_features, hidden_dim).astype(np.float32) * scale_in
        self.b_up = np.zeros(hidden_dim, dtype=np.float32)

        self.W_down = rng.randn(hidden_dim, hidden_dim).astype(np.float32) * scale_out
        self.b_down = np.zeros(hidden_dim, dtype=np.float32)

        self.norm = LayerNormalization(hidden_dim)

    def forward(self, x: np.ndarray, attention_mask: np.ndarray = None) -> np.ndarray:
        if x.ndim == 3:
            x = mean_pooling(x, attention_mask)
        gate = silu(np.matmul(x, self.W_gate) + self.b_gate)
        up = np.matmul(x, self.W_up) + self.b_up
        h = np.matmul(gate * up, self.W_down) + self.b_down
        return self.norm.forward(h)