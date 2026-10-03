import numpy as np
from typing import Dict, Tuple, Optional, List, Any

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
    )

__all__ = [
    "softmax",
    "silu",
    "precompute_rope_frequencies",
    "rotate_half",
    "apply_rope",
    "LayerNormalization",
    "MultiHeadAttention",
    "FeedForwardNetwork",
    "TransformerEncoderLayer",
    "BidirectionalEncoderStack",
    "mean_pooling",
    "mean_pooling_backward",
    "FinalMLP",
]


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


# =====================================================================
# 2. LAYER NORMALIZATION (WITH LOCAL EPHEMERAL CACHING)
# =====================================================================

class LayerNormalization:
    def __init__(self, hidden_dim: int = HIDDEN_DIM, eps: float = LAYER_NORM_EPS):
        self.eps = eps
        self.hidden_dim = hidden_dim
        self.gamma = np.ones(hidden_dim, dtype=np.float32)
        self.beta = np.zeros(hidden_dim, dtype=np.float32)
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        std_inv = 1.0 / np.sqrt(variance + self.eps)
        x_norm = (x - mean) * std_inv
        out = self.gamma * x_norm + self.beta

        # Layer-local ephemeral cache
        self.cache = {
            "x_norm": x_norm,
            "std_inv": std_inv
        }
        return out

    def backward(self, grad_out: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if not self.cache:
            raise RuntimeError("LayerNormalization backward called without preceding forward.")

        x_norm = self.cache["x_norm"]
        std_inv = self.cache["std_inv"]
        N = self.hidden_dim

        grad_gamma = np.sum(grad_out * x_norm, axis=(0, 1))
        grad_beta = np.sum(grad_out, axis=(0, 1))

        # Analytical batch-invariant LayerNorm input gradient
        dx_hat = grad_out * self.gamma
        grad_x = (std_inv / N) * (
            N * dx_hat
            - np.sum(dx_hat, axis=-1, keepdims=True)
            - x_norm * np.sum(dx_hat * x_norm, axis=-1, keepdims=True)
        )

        # Evict cache immediately
        self.cache.clear()

        return grad_x, {"gamma": grad_gamma, "beta": grad_beta}


# =====================================================================
# 3. 8-HEAD ATTENTION (WITH EPHEMERAL CACHING & INVERSE ROPE BACKPROP)
# =====================================================================

class MultiHeadAttention:
    def __init__(self, hidden_dim: int = HIDDEN_DIM, num_heads: int = NUM_HEADS, seed: Optional[int] = None):
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads          # 8 heads
        self.head_dim = hidden_dim // num_heads  # 96 per head
        self.scale = 1.0 / np.sqrt(self.head_dim)
        scale_init = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        rng = np.random.RandomState(seed) if seed is not None else np.random

        self.W_q = (rng.randn(hidden_dim, hidden_dim) * scale_init).astype(np.float32)
        self.b_q = np.zeros(hidden_dim, dtype=np.float32)

        self.W_k = (rng.randn(hidden_dim, hidden_dim) * scale_init).astype(np.float32)
        self.b_k = np.zeros(hidden_dim, dtype=np.float32)

        self.W_v = (rng.randn(hidden_dim, hidden_dim) * scale_init).astype(np.float32)
        self.b_v = np.zeros(hidden_dim, dtype=np.float32)

        self.W_o = (rng.randn(hidden_dim, hidden_dim) * scale_init).astype(np.float32)
        self.b_o = np.zeros(hidden_dim, dtype=np.float32)

        self.cache: Dict[str, np.ndarray] = {}

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))  # [B, 8, L, 96]

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        x = np.transpose(x, (0, 2, 1, 3))
        batch_size, seq_len, _, _ = x.shape
        return x.reshape(batch_size, seq_len, self.hidden_dim)  # [B, L, 768]

    def forward(
        self,
        x: np.ndarray,
        cos: np.ndarray,
        sin: np.ndarray,
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        Q = np.matmul(x, self.W_q) + self.b_q
        K = np.matmul(x, self.W_k) + self.b_k
        V = np.matmul(x, self.W_v) + self.b_v

        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        Q_rot = apply_rope(Q_heads, cos, sin)
        K_rot = apply_rope(K_heads, cos, sin)

        scores = np.matmul(Q_rot, np.transpose(K_rot, (0, 1, 3, 2))) * self.scale

        if attention_mask is not None:
            expanded_mask = attention_mask[:, np.newaxis, np.newaxis, :]
            scores = np.where(expanded_mask == 1.0, scores, -1e9)
        else:
            expanded_mask = None

        attn_weights = softmax(scores, axis=-1)

        context = np.matmul(attn_weights, V_heads)
        merged_context = self._merge_heads(context)
        out = np.matmul(merged_context, self.W_o) + self.b_o

        # Layer-local ephemeral cache
        self.cache = {
            "x": x,
            "merged_context": merged_context,
            "attn_weights": attn_weights,
            "V_heads": V_heads,
            "Q_rot": Q_rot,
            "K_rot": K_rot,
            "cos": cos,
            "sin": sin,
            "expanded_mask": expanded_mask
        }
        return out

    def backward(self, grad_out: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if not self.cache:
            raise RuntimeError("MultiHeadAttention backward called without preceding forward.")

        x = self.cache["x"]
        merged_context = self.cache["merged_context"]
        attn_weights = self.cache["attn_weights"]
        V_heads = self.cache["V_heads"]
        Q_rot = self.cache["Q_rot"]
        K_rot = self.cache["K_rot"]
        cos = self.cache["cos"]
        sin = self.cache["sin"]
        expanded_mask = self.cache["expanded_mask"]

        # 1. Output Linear Projection Gradients
        grad_W_o = np.matmul(np.transpose(merged_context, (0, 2, 1)), grad_out).sum(axis=0)
        grad_b_o = grad_out.sum(axis=(0, 1))
        grad_merged = np.matmul(grad_out, self.W_o.T)

        # 2. Split into 8 Heads
        grad_context = self._split_heads(grad_merged)  # [B, 8, L, 96]

        # 3. Context = Attn @ V
        grad_attn_weights = np.matmul(grad_context, np.transpose(V_heads, (0, 1, 3, 2)))
        grad_V_heads = np.matmul(np.transpose(attn_weights, (0, 1, 3, 2)), grad_context)

        # 4. Softmax Derivative
        sum_grad = np.sum(grad_attn_weights * attn_weights, axis=-1, keepdims=True)
        grad_scores = attn_weights * (grad_attn_weights - sum_grad)

        if expanded_mask is not None:
            grad_scores = np.where(expanded_mask == 1.0, grad_scores, 0.0)

        grad_scores = grad_scores * self.scale

        # 5. Scores = Q_rot @ K_rot.T
        grad_Q_rot = np.matmul(grad_scores, K_rot)
        grad_K_rot = np.matmul(np.transpose(grad_scores, (0, 1, 3, 2)), Q_rot)

        # 6. Inverse RoPE: Rotation transpose is rotation with -sin
        grad_Q_heads = apply_rope(grad_Q_rot, cos, -sin)
        grad_K_heads = apply_rope(grad_K_rot, cos, -sin)

        # 7. Merge Heads
        grad_Q = self._merge_heads(grad_Q_heads)
        grad_K = self._merge_heads(grad_K_heads)
        grad_V = self._merge_heads(grad_V_heads)

        # 8. Input Linear Projections Gradients
        x_T = np.transpose(x, (0, 2, 1))
        grad_W_q = np.matmul(x_T, grad_Q).sum(axis=0)
        grad_b_q = grad_Q.sum(axis=(0, 1))

        grad_W_k = np.matmul(x_T, grad_K).sum(axis=0)
        grad_b_k = grad_K.sum(axis=(0, 1))

        grad_W_v = np.matmul(x_T, grad_V).sum(axis=0)
        grad_b_v = grad_V.sum(axis=(0, 1))

        # 9. Upstream input gradient
        grad_x = (
            np.matmul(grad_Q, self.W_q.T)
            + np.matmul(grad_K, self.W_k.T)
            + np.matmul(grad_V, self.W_v.T)
        )

        # 10. Evict ephemeral cache
        self.cache.clear()

        grads = {
            "W_q": grad_W_q, "b_q": grad_b_q,
            "W_k": grad_W_k, "b_k": grad_b_k,
            "W_v": grad_W_v, "b_v": grad_b_v,
            "W_o": grad_W_o, "b_o": grad_b_o,
        }
        return grad_x, grads


# =====================================================================
# 4. FEED-FORWARD NETWORK (SWIGLU WITH EPHEMERAL CACHING)
# =====================================================================

class FeedForwardNetwork:
    def __init__(self, hidden_dim: int = HIDDEN_DIM, ffn_dim: int = FFN_DIM, seed: Optional[int] = None):
        scale_in = np.sqrt(2.0 / (hidden_dim + ffn_dim))
        scale_out = np.sqrt(2.0 / (ffn_dim + hidden_dim))
        rng = np.random.RandomState(seed) if seed is not None else np.random

        self.W_gate = (rng.randn(hidden_dim, ffn_dim) * scale_in).astype(np.float32)
        self.b_gate = np.zeros(ffn_dim, dtype=np.float32)
        self.W_up = (rng.randn(hidden_dim, ffn_dim) * scale_in).astype(np.float32)
        self.b_up = np.zeros(ffn_dim, dtype=np.float32)
        self.W_down = (rng.randn(ffn_dim, hidden_dim) * scale_out).astype(np.float32)
        self.b_down = np.zeros(hidden_dim, dtype=np.float32)

        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate_lin = np.matmul(x, self.W_gate) + self.b_gate
        sig = 1.0 / (1.0 + np.exp(-np.clip(gate_lin, -SILU_CLIP_BOUND, SILU_CLIP_BOUND)))
        gate_act = gate_lin * sig

        up_lin = np.matmul(x, self.W_up) + self.b_up
        h_mid = gate_act * up_lin
        out = np.matmul(h_mid, self.W_down) + self.b_down

        # Layer-local ephemeral cache
        self.cache = {
            "x": x,
            "gate_lin": gate_lin,
            "sig": sig,
            "gate_act": gate_act,
            "up_lin": up_lin,
            "h_mid": h_mid
        }
        return out

    def backward(self, grad_out: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if not self.cache:
            raise RuntimeError("FeedForwardNetwork backward called without preceding forward.")

        x = self.cache["x"]
        gate_lin = self.cache["gate_lin"]
        sig = self.cache["sig"]
        gate_act = self.cache["gate_act"]
        up_lin = self.cache["up_lin"]
        h_mid = self.cache["h_mid"]

        # Gradients through W_down
        grad_W_down = np.matmul(np.transpose(h_mid, (0, 2, 1)), grad_out).sum(axis=0)
        grad_b_down = grad_out.sum(axis=(0, 1))
        grad_h_mid = np.matmul(grad_out, self.W_down.T)

        grad_up_lin = grad_h_mid * gate_act
        grad_gate_act = grad_h_mid * up_lin

        # Derivative of SiLU: d/dx[x*sig] = sig + x*sig*(1 - sig)
        dsilu = sig + gate_lin * sig * (1.0 - sig)
        grad_gate_lin = grad_gate_act * dsilu

        x_T = np.transpose(x, (0, 2, 1))
        grad_W_gate = np.matmul(x_T, grad_gate_lin).sum(axis=0)
        grad_b_gate = grad_gate_lin.sum(axis=(0, 1))

        grad_W_up = np.matmul(x_T, grad_up_lin).sum(axis=0)
        grad_b_up = grad_up_lin.sum(axis=(0, 1))

        grad_x = np.matmul(grad_gate_lin, self.W_gate.T) + np.matmul(grad_up_lin, self.W_up.T)

        # Evict cache
        self.cache.clear()

        grads = {
            "W_gate": grad_W_gate, "b_gate": grad_b_gate,
            "W_up": grad_W_up, "b_up": grad_b_up,
            "W_down": grad_W_down, "b_down": grad_b_down,
        }
        return grad_x, grads


# =====================================================================
# 5. SINGLE ENCODER LAYER (BACKWARD ORCHESTRATION)
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
        attention_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        attn_out = self.attention.forward(x, cos=cos, sin=sin, attention_mask=attention_mask)
        x = self.norm1.forward(x + attn_out)
        ffn_out = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_out)
        return x

    def backward(self, grad_out: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        # Reverse chain rule through the sub-layers:
        # 1. Backprop through Norm2
        grad_norm2_in, norm2_grads = self.norm2.backward(grad_out)

        # 2. Residual connection 2 branch split
        grad_ffn_out = grad_norm2_in
        grad_norm1_res = grad_norm2_in

        # 3. Backprop through FFN (SwiGLU)
        grad_norm1_ffn, ffn_grads = self.ffn.backward(grad_ffn_out)
        grad_norm1_out = grad_norm1_res + grad_norm1_ffn

        # 4. Backprop through Norm1
        grad_norm1_in, norm1_grads = self.norm1.backward(grad_norm1_out)

        # 5. Residual connection 1 branch split
        grad_attn_out = grad_norm1_in
        grad_x_res = grad_norm1_in

        # 6. Backprop through 8-Head Attention
        grad_x_attn, attn_grads = self.attention.backward(grad_attn_out)
        grad_x = grad_x_res + grad_x_attn

        collected_grads = {
            **{f"attn_{k}": v for k, v in attn_grads.items()},
            **{f"norm1_{k}": v for k, v in norm1_grads.items()},
            **{f"ffn_{k}": v for k, v in ffn_grads.items()},
            **{f"norm2_{k}": v for k, v in norm2_grads.items()},
        }
        return grad_x, collected_grads


# =====================================================================
# 6. FULL 12-LAYER BIDIRECTIONAL ENCODER STACK
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

    def forward(self, x: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        seq_len = x.shape[1]
        cos, sin = precompute_rope_frequencies(self.head_dim, seq_len)
        hidden_states = x
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, cos=cos, sin=sin, attention_mask=attention_mask)
        return hidden_states

    def backward(self, grad_out: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        all_grads = {}
        grad_flow = grad_out

        # Reverse propagation through all layers
        for layer_idx in reversed(range(self.num_layers)):
            layer = self.layers[layer_idx]
            grad_flow, layer_grads = layer.backward(grad_flow)
            for param_name, param_grad in layer_grads.items():
                all_grads[f"layer_{layer_idx}_{param_name}"] = param_grad

        return grad_flow, all_grads

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Dynamically exports all weight and bias arrays across layers."""
        weights: Dict[str, np.ndarray] = {}
        for i, layer in enumerate(self.layers):
            for sub_name, sub_module in [
                ("attn", layer.attention),
                ("norm1", layer.norm1),
                ("ffn", layer.ffn),
                ("norm2", layer.norm2)
            ]:
                for attr_name, attr_val in vars(sub_module).items():
                    if isinstance(attr_val, np.ndarray) and not attr_name.startswith("_") and attr_name != "cache":
                        # Standard key format matching historical checkpoint keys:
                        # e.g., enc_l0_W_q, enc_l0_norm1_gamma, enc_l0_W_gate, enc_l0_norm2_beta
                        key = f"enc_l{i}_{sub_name}_{attr_name}" if sub_name in ("norm1", "norm2") else f"enc_l{i}_{attr_name}"
                        weights[key] = attr_val.copy()
        return weights

    def set_weights(self, weights: Dict[str, Any]) -> None:
        """Dynamically loads all weights into sub-modules matching attribute names."""
        for i, layer in enumerate(self.layers):
            for sub_name, sub_module in [
                ("attn", layer.attention),
                ("norm1", layer.norm1),
                ("ffn", layer.ffn),
                ("norm2", layer.norm2)
            ]:
                for attr_name in list(vars(sub_module).keys()):
                    if attr_name.startswith("_") or attr_name == "cache":
                        continue
                    # Check both standard and namespaced key formats
                    candidate_keys = [
                        f"enc_l{i}_{attr_name}",
                        f"enc_l{i}_{sub_name}_{attr_name}",
                    ]
                    for key in candidate_keys:
                        if key in weights:
                            setattr(sub_module, attr_name, weights[key].astype(np.float32))
                            break


# =====================================================================
# 7. POOLING & SITUATION FINAL MLP
# =====================================================================

def mean_pooling(token_embeddings: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
    if token_embeddings.ndim == 2:
        return token_embeddings
    if attention_mask is not None:
        expanded_mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * expanded_mask, axis=1)
        sum_mask = np.clip(np.sum(expanded_mask, axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask
    else:
        return np.mean(token_embeddings, axis=1)


def mean_pooling_backward(
    grad_pooled: np.ndarray,
    seq_len: int,
    attention_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Distributes pooled gradient back to individual token positions."""
    if attention_mask is not None:
        expanded_mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_mask = np.clip(np.sum(expanded_mask, axis=1, keepdims=True), a_min=1e-9, a_max=None)
        grad_tokens = (grad_pooled[:, np.newaxis, :] * expanded_mask) / sum_mask
    else:
        grad_tokens = np.repeat(grad_pooled[:, np.newaxis, :] / seq_len, seq_len, axis=1)
    return grad_tokens.astype(np.float32)


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
        self.W_gate = (rng.randn(in_features, hidden_dim) * scale_in).astype(np.float32)
        self.b_gate = np.zeros(hidden_dim, dtype=np.float32)
        self.W_up = (rng.randn(in_features, hidden_dim) * scale_in).astype(np.float32)
        self.b_up = np.zeros(hidden_dim, dtype=np.float32)
        self.W_down = (rng.randn(hidden_dim, hidden_dim) * scale_out).astype(np.float32)
        self.b_down = np.zeros(hidden_dim, dtype=np.float32)
        self.norm = LayerNormalization(hidden_dim)
        self.cache: Dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray, attention_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if x.ndim == 3:
            x = mean_pooling(x, attention_mask)

        gate_lin = np.matmul(x, self.W_gate) + self.b_gate
        sig = 1.0 / (1.0 + np.exp(-np.clip(gate_lin, -SILU_CLIP_BOUND, SILU_CLIP_BOUND)))
        gate_act = gate_lin * sig

        up_lin = np.matmul(x, self.W_up) + self.b_up
        h_mid = gate_act * up_lin
        h = np.matmul(h_mid, self.W_down) + self.b_down
        out = self.norm.forward(h)

        # Cache for backward
        self.cache = {
            "x": x,
            "gate_lin": gate_lin,
            "sig": sig,
            "gate_act": gate_act,
            "up_lin": up_lin,
            "h_mid": h_mid
        }
        return out

    def backward(self, grad_out: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        if not self.cache:
            raise RuntimeError("FinalMLP backward called without preceding forward.")

        x = self.cache["x"]
        gate_lin = self.cache["gate_lin"]
        sig = self.cache["sig"]
        gate_act = self.cache["gate_act"]
        up_lin = self.cache["up_lin"]
        h_mid = self.cache["h_mid"]

        # Backprop through LayerNorm
        grad_h, norm_grads = self.norm.backward(grad_out)

        # Backprop through SwiGLU
        grad_W_down = np.matmul(h_mid.T, grad_h)
        grad_b_down = np.sum(grad_h, axis=0)
        grad_h_mid = np.matmul(grad_h, self.W_down.T)

        grad_up_lin = grad_h_mid * gate_act
        grad_gate_act = grad_h_mid * up_lin

        dsilu = sig + gate_lin * sig * (1.0 - sig)
        grad_gate_lin = grad_gate_act * dsilu

        grad_W_gate = np.matmul(x.T, grad_gate_lin)
        grad_b_gate = np.sum(grad_gate_lin, axis=0)

        grad_W_up = np.matmul(x.T, grad_up_lin)
        grad_b_up = np.sum(grad_up_lin, axis=0)

        grad_x = np.matmul(grad_gate_lin, self.W_gate.T) + np.matmul(grad_up_lin, self.W_up.T)

        self.cache.clear()

        grads = {
            "mlp_W_gate": grad_W_gate, "mlp_b_gate": grad_b_gate,
            "mlp_W_up": grad_W_up, "mlp_b_up": grad_b_up,
            "mlp_W_down": grad_W_down, "mlp_b_down": grad_b_down,
            "mlp_norm_gamma": norm_grads["gamma"], "mlp_norm_beta": norm_grads["beta"]
        }
        return grad_x, grads