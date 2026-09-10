#===============All imports are placed here===================
import torch
import torch.nn as nn
import math
try:
    from src.config import default_model_config
except (ImportError, ModuleNotFoundError):
    from config import default_model_config
#=============================================================

#=============Mathematic execution functions==================
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv_heads, s, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv_heads, n_rep, s, head_dim)
        .reshape(b, n_kv_heads * n_rep, s, head_dim)
    )

def build_prefix_causal_mask(total_len: int, prefix_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    mask = torch.full((total_len, total_len), float('-inf'), device=device, dtype=dtype)
    tril = torch.tril(torch.ones((total_len, total_len), device=device, dtype=torch.bool))
    mask = mask.masked_fill(tril, 0.0)

    if prefix_len > 0:
        mask[:prefix_len, :prefix_len] = 0.0

    return mask.unsqueeze(0).unsqueeze(0)
#==============================================================

#===================RoPE and GQA implementation========================
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

#=======================Attention Mechanisms========================
class CausalSelfAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = dropout

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        is_causal = (attn_mask is None)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=self.dropout if self.training else 0.0
        )

class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_model=default_model_config.ninp,
        nhead=default_model_config.nhead,
        n_kv_heads=default_model_config.n_kv_heads,
        dropout=default_model_config.dropout
    ):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        assert nhead % n_kv_heads == 0, f"nhead ({nhead}) must be divisible by n_kv_heads ({n_kv_heads})"
        self.d_model = d_model
        self.nhead = nhead
        self.n_kv_heads = n_kv_heads
        self.num_kv_groups = nhead // n_kv_heads
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.inner_attn = CausalSelfAttention(dropout=dropout)

    def forward(self, x, cos, sin, attn_mask=None, prefix_len=0):
        B, S, D = x.shape

        q = self.q_proj(x).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if prefix_len > 0:
            text_len = S - prefix_len
            q_prefix = q[:, :, :prefix_len, :]
            k_prefix = k[:, :, :prefix_len, :]

            if text_len > 0:
                q_text = apply_rotary_pos_emb(q[:, :, prefix_len:, :], cos[:, :, :text_len, :], sin[:, :, :text_len, :])
                k_text = apply_rotary_pos_emb(k[:, :, prefix_len:, :], cos[:, :, :text_len, :], sin[:, :, :text_len, :])
                q = torch.cat([q_prefix, q_text], dim=2)
                k = torch.cat([k_prefix, k_text], dim=2)
            else:
                q = q_prefix
                k = k_prefix
        else:
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        out = self.inner_attn(q, k, v, attn_mask=attn_mask)

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)
#==============================================================

#=======================Transformer Block and Model========================
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model=default_model_config.ninp,
        nhead=default_model_config.nhead,
        n_kv_heads=default_model_config.n_kv_heads,
        nhid=default_model_config.nhid,
        dropout=default_model_config.dropout
    ):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = GroupedQueryAttention(d_model=d_model, nhead=nhead, n_kv_heads=n_kv_heads, dropout=dropout)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, nhid, bias=False),
            nn.GELU(),
            nn.Linear(nhid, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x, cos, sin, attn_mask=None, prefix_len=0):
        x = x + self.attn(self.ln_1(x), cos, sin, attn_mask=attn_mask, prefix_len=prefix_len)
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken=default_model_config.ntoken,
        ninp=default_model_config.ninp,
        nhead=default_model_config.nhead,
        n_kv_heads=default_model_config.n_kv_heads,
        nhid=default_model_config.nhid,
        nlayers=default_model_config.nlayers,
        dropout=default_model_config.dropout,
        max_seq_len=default_model_config.max_seq_len
    ):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.ninp = ninp
        self.ntoken = ntoken
        self.nhead = nhead
        self.n_kv_heads = n_kv_heads

        self.encoder = nn.Embedding(ntoken, ninp)
        self.rotary_emb = RotaryEmbedding(dim=ninp // nhead, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=ninp, nhead=nhead, n_kv_heads=n_kv_heads, nhid=nhid, dropout=dropout)
            for _ in range(nlayers)
        ])
        self.norm = nn.LayerNorm(ninp)
        self.decoder = nn.Linear(ninp, ntoken, bias=False)
        self.decoder.weight = self.encoder.weight #Weight tying between the encoder and decoder to reduce the number of parameters and improve generalization.

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.orthogonal_(p)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, src=None, inputs_embeds=None, prefix_len=0):
        if inputs_embeds is not None:
            x = inputs_embeds
            B, S, D = x.shape
        elif src is not None:
            B, S = src.shape
            x = self.encoder(src) * math.sqrt(self.ninp)
        else:
            raise ValueError("Either src (token IDs) or inputs_embeds must be provided.")

        x = self.dropout(x)

        if prefix_len > 0:
            text_len = S - prefix_len
            cos, sin = self.rotary_emb(x, max(text_len, 1))
            attn_mask = build_prefix_causal_mask(S, prefix_len, device=x.device, dtype=x.dtype)
        else:
            cos, sin = self.rotary_emb(x, S)
            attn_mask = None

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask=attn_mask, prefix_len=prefix_len)

        x = self.norm(x)
        logits = self.decoder(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx=None,
        inputs_embeds=None,
        prefix_len=0,
        max_new_tokens=250,
        temperature=0.5,
        top_k=30,
        top_p=0.85,
        repetition_penalty=1.2,
        eos_token_id=None
    ):
        if idx is None and inputs_embeds is None:
            raise ValueError("Either idx or inputs_embeds must be provided to generate")

        if inputs_embeds is not None and prefix_len == 0:
            prefix_len = inputs_embeds.size(1)

        generated_ids = []
        current_embeds = inputs_embeds
        current_idx = idx

        for _ in range(max_new_tokens):
            if current_embeds is not None:
                max_ctx = self.rotary_emb.max_seq_len
                cond_embeds = current_embeds if current_embeds.size(1) <= max_ctx else current_embeds[:, -max_ctx:]
                logits = self(inputs_embeds=cond_embeds, prefix_len=prefix_len)
            else:
                idx_cond = current_idx if current_idx.size(1) <= 8192 else current_idx[:, -8192:]
                logits = self(src=idx_cond, prefix_len=0)

            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if repetition_penalty != 1.0 and (current_idx is not None or generated_ids):
                penalty_tokens = set(current_idx[0, -64:].tolist()) if current_idx is not None else set(generated_ids[-64:])
                for b in range(logits.size(0)):
                    for token_id in penalty_tokens:
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            generated_ids.append(idx_next.item() if idx_next.numel() == 1 else idx_next[0, 0].item())

            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

            if current_embeds is not None:
                next_embed = self.encoder(idx_next) * math.sqrt(self.ninp)
                current_embeds = torch.cat((current_embeds, next_embed), dim=1)
            else:
                current_idx = torch.cat((current_idx, idx_next), dim=1)

        if idx is not None:
            return current_idx
        else:
            dev = inputs_embeds.device if inputs_embeds is not None else torch.device("cpu")
            return torch.tensor([generated_ids], device=dev)

#===============================================================


#=======================Testing the model========================

if __name__ == "__main__":
    model = TransformerModel()
    unique_params = set(model.parameters())
    total_params = sum(p.numel() for p in unique_params)
    trainable_params = sum(p.numel() for p in unique_params if p.requires_grad)
    print(f"Total Parameters (unique):     {total_params:,}")
    dummy_input = torch.randint(0, model.ntoken, (4, 32)) 
    dummy_output = model(dummy_input)
    print(f"Output shape: {dummy_output.shape}")
