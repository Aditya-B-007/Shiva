#===============All imports are placed here===================
import torch
import torch.nn as nn
import math
from typing import Optional

try:
    from src.interfaces import ITextGenerator
    from src.config import (
        default_model_config,
        ROPE_BASE,
        MAX_GENERATE_CONTEXT,
        REPETITION_PENALTY_WINDOW,
        TEMPERATURE_EPSILON,
    )
except (ImportError, ModuleNotFoundError):
    from interfaces import ITextGenerator
    from config import (
        default_model_config,
        ROPE_BASE,
        MAX_GENERATE_CONTEXT,
        REPETITION_PENALTY_WINDOW,
        TEMPERATURE_EPSILON,
    )
#=============================================================

#=============Mathematic execution functions==================
def rotate_half(tensor):
    x1 = tensor[..., : tensor.shape[-1] // 2]
    x2 = tensor[..., tensor.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(queryOrKey, cos, sin):
    return (queryOrKey * cos) + (rotate_half(queryOrKey) * sin)

def repeat_kv(kvTensor: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return kvTensor
    b, n_kv_heads, s, head_dim = kvTensor.shape
    return (
        kvTensor[:, :, None, :, :]
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
class RoPEEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=8192, base=ROPE_BASE):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len = seq_len
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, dim // 2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, tensor, seq_len):
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cached[:, :, :seq_len, :].to(dtype=tensor.dtype, device=tensor.device),
            self.sin_cached[:, :, :seq_len, :].to(dtype=tensor.dtype, device=tensor.device),
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
        d_model=None,
        nhead=None,
        n_kv_heads=None,
        dropout=None
    ):
        super().__init__()
        d_model = d_model if d_model is not None else default_model_config.ninp
        nhead = nhead if nhead is not None else default_model_config.nhead
        n_kv_heads = n_kv_heads if n_kv_heads is not None else default_model_config.n_kv_heads
        dropout = dropout if dropout is not None else default_model_config.dropout

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
        d_model=None,
        nhead=None,
        n_kv_heads=None,
        nhid=None,
        dropout=None
    ):
        super().__init__()
        d_model = d_model if d_model is not None else default_model_config.ninp
        nhead = nhead if nhead is not None else default_model_config.nhead
        n_kv_heads = n_kv_heads if n_kv_heads is not None else default_model_config.n_kv_heads
        nhid = nhid if nhid is not None else default_model_config.nhid
        dropout = dropout if dropout is not None else default_model_config.dropout

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
        ntoken=None,
        ninp=None,
        nhead=None,
        n_kv_heads=None,
        nhid=None,
        nlayers=None,
        dropout=None,
        max_seq_len=None
    ):
        super(TransformerModel, self).__init__()
        ntoken = ntoken if ntoken is not None else default_model_config.ntoken
        ninp = ninp if ninp is not None else default_model_config.ninp
        nhead = nhead if nhead is not None else default_model_config.nhead
        n_kv_heads = n_kv_heads if n_kv_heads is not None else default_model_config.n_kv_heads
        nhid = nhid if nhid is not None else default_model_config.nhid
        nlayers = nlayers if nlayers is not None else default_model_config.nlayers
        dropout = dropout if dropout is not None else default_model_config.dropout
        max_seq_len = max_seq_len if max_seq_len is not None else default_model_config.max_seq_len

        self.model_type = 'Transformer'
        self.ninp = ninp
        self.ntoken = ntoken
        self.nhead = nhead
        self.n_kv_heads = n_kv_heads

        self.encoder = nn.Embedding(ntoken, ninp)
        self.rotary_emb = RoPEEmbedding(dim=ninp // nhead, max_seq_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model=ninp, nhead=nhead, n_kv_heads=n_kv_heads, nhid=nhid, dropout=dropout)
            for _ in range(nlayers)
        ])
        self.norm = nn.LayerNorm(ninp)
        self.decoder = nn.Linear(ninp, ntoken, bias=False)
        self.decoder.weight = self.encoder.weight  # Weight tying: reduces parameters and improves generalisation.

        self._generator: Optional[TextGenerator] = None
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.orthogonal_(p)
        for m in self.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.encoder

    def resize_token_embeddings(self, new_num_tokens: int):
        if new_num_tokens == self.ntoken:
            return
        old_embeddings = self.encoder
        new_embeddings = nn.Embedding(new_num_tokens, self.ninp, device=old_embeddings.weight.device)
        nn.init.orthogonal_(new_embeddings.weight)
        n_copy = min(self.ntoken, new_num_tokens)
        with torch.no_grad():
            new_embeddings.weight[:n_copy] = old_embeddings.weight[:n_copy]
        self.encoder = new_embeddings
        self.decoder = nn.Linear(self.ninp, new_num_tokens, bias=False, device=old_embeddings.weight.device)
        self.decoder.weight = self.encoder.weight
        self.ntoken = new_num_tokens

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
        if self._generator is None:
            self._generator = TextGenerator(self)

        return self._generator.generateTokens(
            tokenIndices=idx,
            inputsEmbeds=inputs_embeds,
            prefixLength=prefix_len,
            maxNewTokens=max_new_tokens,
            temperature=temperature,
            topK=top_k,
            topP=top_p,
            repetitionPenalty=repetition_penalty,
            eosTokenId=eos_token_id
        )

#===============================================================

#===================Decoupled Text Generator====================
class TextGenerator(ITextGenerator):
    def __init__(self, languageModel: TransformerModel):
        self.languageModel = languageModel

    @torch.no_grad()
    def generateTokens(
        self,
        tokenIndices: Optional[torch.Tensor] = None,
        inputsEmbeds: Optional[torch.Tensor] = None,
        prefixLength: int = 0,
        maxNewTokens: int = 250,
        temperature: float = 0.5,
        topK: int = 30,
        topP: float = 0.85,
        repetitionPenalty: float = 1.2,
        eosTokenId: Optional[int] = None
    ) -> torch.Tensor:
        if tokenIndices is None and inputsEmbeds is None:
            raise ValueError("Either tokenIndices or inputsEmbeds must be provided to generateTokens")

        if inputsEmbeds is not None and prefixLength == 0:
            prefixLength = inputsEmbeds.size(1)

        generatedIds = []
        generatedTokens = None
        currentEmbeds = inputsEmbeds
        currentIdx = tokenIndices

        for _ in range(maxNewTokens):
            if currentEmbeds is not None:
                maxCtx = self.languageModel.rotary_emb.max_seq_len
                condEmbeds = currentEmbeds if currentEmbeds.size(1) <= maxCtx else currentEmbeds[:, -maxCtx:]
                logits = self.languageModel(inputs_embeds=condEmbeds, prefix_len=prefixLength)
            else:
                idxCond = currentIdx if currentIdx.size(1) <= MAX_GENERATE_CONTEXT else currentIdx[:, -MAX_GENERATE_CONTEXT:]
                logits = self.languageModel(src=idxCond, prefix_len=0)

            logits = logits[:, -1, :] / max(temperature, TEMPERATURE_EPSILON)
            if repetitionPenalty != 1.0 and (currentIdx is not None or generatedIds):
                penaltyTokens = set(currentIdx[0, -REPETITION_PENALTY_WINDOW:].tolist()) if currentIdx is not None else set(generatedIds[-REPETITION_PENALTY_WINDOW:])
                for b in range(logits.size(0)):
                    for tokenId in penaltyTokens:
                        if logits[b, tokenId] > 0:
                            logits[b, tokenId] /= repetitionPenalty
                        else:
                            logits[b, tokenId] *= repetitionPenalty

            if topK is not None:
                v, _ = torch.topk(logits, min(topK, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if topP is not None and topP < 1.0:
                sortedLogits, sortedIndices = torch.sort(logits, descending=True)
                cumulativeProbs = torch.cumsum(torch.softmax(sortedLogits, dim=-1), dim=-1)
                sortedIndicesToRemove = cumulativeProbs > topP
                sortedIndicesToRemove[..., 1:] = sortedIndicesToRemove[..., :-1].clone()
                sortedIndicesToRemove[..., 0] = 0
                indicesToRemove = sortedIndicesToRemove.scatter(1, sortedIndices, sortedIndicesToRemove)
                logits[indicesToRemove] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            idxNext = torch.multinomial(probs, num_samples=1)

            generatedIds.append(idxNext.item() if idxNext.numel() == 1 else idxNext[0, 0].item())

            if eosTokenId is not None and (idxNext == eosTokenId).all():
                break

            if currentEmbeds is not None:
                generatedTokens = idxNext if generatedTokens is None else torch.cat((generatedTokens, idxNext), dim=1)
                nextEmbed = self.languageModel.encoder(idxNext) * math.sqrt(self.languageModel.ninp)
                currentEmbeds = torch.cat((currentEmbeds, nextEmbed), dim=1)
            else:
                currentIdx = torch.cat((currentIdx, idxNext), dim=1)

        if tokenIndices is not None:
            return currentIdx
        else:
            return generatedTokens if generatedTokens is not None else torch.empty((inputsEmbeds.size(0), 0), dtype=torch.long, device=inputsEmbeds.device)

#===============================================================
