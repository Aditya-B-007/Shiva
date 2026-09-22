import numpy as np

# =====================================================================
# 1. CORE MATHEMATICAL & ROPE PRIMITIVES
# =====================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def precompute_rope_frequencies(head_dim: int, seq_len: int, theta: float = 10000.0):
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
# 2. MULTI-HEAD ATTENTION 
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
    def __init__(self, hidden_dim: int = 768, ffn_dim: int = 3072):
        scale_in = np.sqrt(2.0 / (hidden_dim + ffn_dim))
        scale_out = np.sqrt(2.0 / (ffn_dim + hidden_dim))
        
        self.W_gate = np.random.randn(hidden_dim, ffn_dim).astype(np.float32) * scale_in
        self.b_gate = np.zeros(ffn_dim, dtype=np.float32)

        self.W_up = np.random.randn(hidden_dim, ffn_dim).astype(np.float32) * scale_in
        self.b_up = np.zeros(ffn_dim, dtype=np.float32)

        self.W_down = np.random.randn(ffn_dim, hidden_dim).astype(np.float32) * scale_out
        self.b_down = np.zeros(hidden_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = silu(np.matmul(x, self.W_gate) + self.b_gate)
        up = np.matmul(x, self.W_up) + self.b_up
        return np.matmul(gate * up, self.W_down) + self.b_down


# =====================================================================
# 4. SINGLE ENCODER LAYER
# =====================================================================

class TransformerEncoderLayer:
    def __init__(self, hidden_dim: int = 768, num_heads: int = 12, ffn_dim: int = 3072):
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = LayerNormalization(hidden_dim)
        self.ffn = FeedForwardNetwork(hidden_dim, ffn_dim)
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
        num_layers: int = 12,
        hidden_dim: int = 768,
        num_heads: int = 12,
        ffn_dim: int = 3072
    ):
        self.num_layers = num_layers
        self.head_dim = hidden_dim // num_heads
        self.layers = [
            TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ]

    def forward(self, x: np.ndarray, attention_mask: np.ndarray = None) -> np.ndarray:
        seq_len = x.shape[1]
        cos, sin = precompute_rope_frequencies(self.head_dim, seq_len)

        hidden_states = x
        for layer in self.layers:
            hidden_states = layer.forward(hidden_states, cos=cos, sin=sin, attention_mask=attention_mask)
        return hidden_states


# =====================================================================
# 6. SITUATION / FINAL MLP 
# =====================================================================

class FinalMLP:
    def __init__(self, in_features: int = 768, hidden_dim: int = 512):
        scale_in = np.sqrt(2.0 / (in_features + hidden_dim))
        scale_out = np.sqrt(2.0 / (hidden_dim + hidden_dim))

        self.W_gate = np.random.randn(in_features, hidden_dim).astype(np.float32) * scale_in
        self.b_gate = np.zeros(hidden_dim, dtype=np.float32)

        self.W_up = np.random.randn(in_features, hidden_dim).astype(np.float32) * scale_in
        self.b_up = np.zeros(hidden_dim, dtype=np.float32)

        self.W_down = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale_out
        self.b_down = np.zeros(hidden_dim, dtype=np.float32)

        self.norm = LayerNormalization(hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        gate = silu(np.matmul(x, self.W_gate) + self.b_gate)
        up = np.matmul(x, self.W_up) + self.b_up
        h = np.matmul(gate * up, self.W_down) + self.b_down
        return self.norm.forward(h)