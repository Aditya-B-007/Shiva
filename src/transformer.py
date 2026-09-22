import numpy as np

# =====================================================================
# 1. CORE MATHEMATICAL PRIMITIVES
# =====================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) approximation: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))"""
    sqrt_2_over_pi = np.sqrt(2.0 / np.pi)
    cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x + 0.044715 * np.power(x, 3))))
    return x * cdf


class LayerNormalization:
    """
    Layer Normalization over the final hidden dimension:
    y = ((x - mean) / sqrt(var + eps)) * gamma + beta
    """
    def __init__(self, hidden_dim: int = 768, eps: float = 1e-12):
        self.eps = eps
        self.gamma = np.ones(hidden_dim, dtype=np.float32)
        self.beta = np.zeros(hidden_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = np.mean(x, axis=-1, keepdims=True)
        variance = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(variance + self.eps)
        return self.gamma * x_norm + self.beta


# =====================================================================
# 2. MULTI-HEAD BIDIRECTIONAL SELF-ATTENTION
# =====================================================================

class MultiHeadAttention:
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12):
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads  # 768 // 12 = 64
        self.scale = 1.0 / np.sqrt(self.head_dim)

        scale_init = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        
        self.W_q = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale_init
        self.b_q = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W_k = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale_init
        self.b_k = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W_v = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale_init
        self.b_v = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W_o = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale_init
        self.b_o = np.zeros(hidden_dim, dtype=np.float32)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return np.transpose(x, (0, 2, 1, 3))

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        x = np.transpose(x, (0, 2, 1, 3))
        batch_size, seq_len, _, _ = x.shape
        return x.reshape(batch_size, seq_len, self.hidden_dim)

    def forward(self, x: np.ndarray, attention_mask: np.ndarray = None) -> np.ndarray:
        Q = np.matmul(x, self.W_q) + self.b_q
        K = np.matmul(x, self.W_k) + self.b_k
        V = np.matmul(x, self.W_v) + self.b_v

        Q_heads = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        scores = np.matmul(Q_heads, np.transpose(K_heads, (0, 1, 3, 2))) * self.scale

        if attention_mask is not None:
            expanded_mask = attention_mask[:, np.newaxis, np.newaxis, :]     
            scores = np.where(expanded_mask == 1, scores, -1e9)
        attn_weights = softmax(scores, axis=-1)
        context = np.matmul(attn_weights, V_heads)
        merged_context = self._merge_heads(context)

        return np.matmul(merged_context, self.W_o) + self.b_o


# =====================================================================
# 3. FEED-FORWARD NETWORK (FFN)
# =====================================================================

class FeedForwardNetwork:
    """
    Two-layer Position-wise Feed-Forward Network:
    FFN(x) = GELU(x * W1 + b1) * W2 + b2
    """
    def __init__(self, hidden_dim: int = 768, ffn_dim: int = 3072):
        scale_1 = np.sqrt(2.0 / (hidden_dim + ffn_dim))
        scale_2 = np.sqrt(2.0 / (ffn_dim + hidden_dim))
        
        self.W1 = np.random.randn(hidden_dim, ffn_dim).astype(np.float32) * scale_1
        self.b1 = np.zeros(ffn_dim, dtype=np.float32)

        self.W2 = np.random.randn(ffn_dim, hidden_dim).astype(np.float32) * scale_2
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        h1 = np.matmul(x, self.W1) + self.b1
        h1_act = gelu(h1)
        return np.matmul(h1_act, self.W2) + self.b2


# =====================================================================
# 4. SINGLE ENCODER LAYER (ATTENTION + FFN + RESIDUALS + NORMS)
# =====================================================================

class TransformerEncoderLayer:
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12, ffn_dim: int = 3072):
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = LayerNormalization(hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_dim, ffn_dim)
        self.norm2 = LayerNormalization(hidden_dim)

    def forward(self, x: np.ndarray, attention_mask: np.ndarray = None) -> np.ndarray:
        attn_out = self.attention.forward(x, attention_mask=attention_mask)
        x = self.norm1.forward(x + attn_out) #Skip attention is added to remove the problem of sparsity.

        ffn_out = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_out)

        return x


# =====================================================================
# 5. THE FULL 12-LAYER BIDIRECTIONAL ENCODER STACK (~85M PARAMS)
# =====================================================================

class BidirectionalEncoderStack:
    def __init__(
        self,
        num_layers: int = 12,
        hidden_dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072
    ):
        self.num_layers = num_layers
        self.layers = [
            TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ]

    def forward(self, x: np.ndarray, attention_mask: np.ndarray = None) -> np.ndarray:
        hidden_states = x
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer.forward(hidden_states, attention_mask=attention_mask)
        return hidden_states