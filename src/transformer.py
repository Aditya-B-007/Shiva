import torch
import torch.nn as nn
import math

# Additions implemented:
# 1. Rotary Position Embeddings (RoPE) applied to Q and K for strong length extrapolation in reasoning.
# 2. FlashAttention / Scaled Dot-Product Attention (SDPA) with causal masking.
# 3. Pre-Layer Normalization architecture with final LayerNorm before the LM head.
# 4. Weight tying between token embedding and output projection matrix.
# 5. Orthogonal weight initialization.
# 6. GELU activation function in the feed-forward network.
# 7. Standard batch-first format (batch_size, seq_len).

def rotate_half(x):
    """Rotates half the hidden dimensions of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    """Applies Rotary Position Embedding to Query or Key tensors."""
    # x: (batch_size, num_heads, seq_len, head_dim)
    # cos, sin: (1, 1, seq_len, head_dim)
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len):
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device),
            self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device),
        )

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=1024, nhead=16, dropout=0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # PyTorch SDPA uses FlashAttention where supported with causal masking
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self, d_model=1024, nhead=16, nhid=4096, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, nhid, bias=False),
            nn.GELU(),
            nn.Linear(nhid, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin):
        # Pre-LN attention block with residual connection
        x = x + self.attn(self.ln_1(x), cos, sin)
        # Pre-LN MLP block with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(self, ntoken=50257, ninp=1024, nhead=16, nhid=4096, nlayers=20, dropout=0.1, max_seq_len=8192):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.ninp = ninp
        self.ntoken = ntoken
        self.nhead = nhead

        self.encoder = nn.Embedding(ntoken, ninp)
        self.rotary_emb = RotaryEmbedding(dim=ninp // nhead, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=ninp, nhead=nhead, nhid=nhid, dropout=dropout)
            for _ in range(nlayers)
        ])

        # Pre-LN final LayerNorm
        self.norm = nn.LayerNorm(ninp)

        # Decoder LM Head
        self.decoder = nn.Linear(ninp, ntoken, bias=False)

        # Weight tying
        self.decoder.weight = self.encoder.weight

        self._init_weights()

    def _init_weights(self):
        # Orthogonal initialization for 2D+ weight matrices
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.orthogonal_(p)
        
        # LayerNorm initialization
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, src):
        # src: (batch_size, seq_len)
        B, S = src.shape

        # Token embeddings scaled by sqrt(d_model)
        x = self.encoder(src) * math.sqrt(self.ninp)
        x = self.dropout(x)

        # Compute RoPE cos & sin once for the current sequence length
        cos, sin = self.rotary_emb(x, S)

        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, cos, sin)

        x = self.norm(x)
        logits = self.decoder(x)
        return logits

if __name__ == "__main__":
    # Test batch_first: (batch_size=4, seq_len=32)
    model = TransformerModel(ntoken=50257, ninp=1024, nhead=16, nhid=4096, nlayers=20)
    
    # Deduplicate tied parameters when counting
    unique_params = set(model.parameters())
    total_params = sum(p.numel() for p in unique_params)
    trainable_params = sum(p.numel() for p in unique_params if p.requires_grad)
    
    print(f"Total Parameters (unique):     {total_params:,}")
    print(f"Trainable Parameters (unique): {trainable_params:,}")
    print(f"Target Parameter Fit:          {total_params / 1_000_000:.2f} Million Parameters")

    # Shape: (batch_size=4, seq_len=32)
    dummy_input = torch.randint(0, 50257, (4, 32)) 
    dummy_output = model(dummy_input)
    print(f"Output shape successfully verified: {dummy_output.shape}")
