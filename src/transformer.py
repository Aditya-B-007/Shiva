import torch
import torch.nn as nn
import math

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
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
    def __init__(self, ntoken=9437, ninp=512, nhead=8, nhid=2048, nlayers=8, dropout=0.1, max_seq_len=8192):
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

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=250, temperature=0.5, top_k=30, top_p=0.85, repetition_penalty=1.2, eos_token_id=None):
        """
        Improved generation loop with repetition penalty and nucleus sampling.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= 8192 else idx[:, -8192:]
            
            logits = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Apply repetition penalty to recently generated tokens
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    recent_tokens = set(idx[b, -64:].tolist())
                    for token_id in recent_tokens:
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            # Top-K filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-P (Nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep first token above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

if __name__ == "__main__":
    # Test batch_first: (batch_size=4, seq_len=32)
    model = TransformerModel(ntoken=9437, ninp=512, nhead=8, nhid=2048, nlayers=8)
    
    # Deduplicate tied parameters when counting
    unique_params = set(model.parameters())
    total_params = sum(p.numel() for p in unique_params)
    trainable_params = sum(p.numel() for p in unique_params if p.requires_grad)
    
    print(f"Total Parameters (unique):     {total_params:,}")
    print(f"Trainable Parameters (unique): {trainable_params:,}")
    print(f"Target Parameter Fit:          {total_params / 1_000_000:.2f} Million Parameters")

    # Shape: (batch_size=4, seq_len=32)
    dummy_input = torch.randint(0, 9437, (4, 32)) 
    dummy_output = model(dummy_input)
    print(f"Output shape successfully verified: {dummy_output.shape}")
