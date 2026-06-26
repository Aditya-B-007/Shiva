from __future__ import annotations
import math
from typing import Union
import torch
import torch.nn as nn

class GateHyperNetwork(nn.Module):

    def __init__(self, d_model: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or d_model // 2
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )
        # Zero-init output layer → gates start at sigmoid(0) = 0.5.
        nn.init.zeros_(self.net[-1].weight)  # type: ignore[arg-type]
        nn.init.zeros_(self.net[-1].bias)    # type: ignore[arg-type]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class TransformerEncoderBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # --- Attention projections ---
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # --- Feed-forward network ---
        ff_dim = d_model * 4
        self.ff_net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )

        # --- Layer norms (Pre-LN style compatible with post-LN placement) ---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # --- Dynamic gating networks ---
        self.attn_gate_net = GateHyperNetwork(d_model)
        self.ff_gate_net = GateHyperNetwork(d_model)

        # --- Emotion-to-head routing ---
        # A scalar bias added uniformly to all attention logits cancels in
        # softmax (shift invariance). Instead, we learn a per-head projection
        # from the scalar valence so each head gets its own bias, making the
        # emotional signal non-uniform and therefore meaningful post-softmax.
        # Zero-init ensures the model starts in a neutral emotional state.
        self.emotion_head_router = nn.Linear(1, num_heads)
        nn.init.zeros_(self.emotion_head_router.weight)
        nn.init.zeros_(self.emotion_head_router.bias)

    # ------------------------------------------------------------------
    # Internal: multi-head attention
    # ------------------------------------------------------------------

    def _multi_head_attention(
        self,
        x: torch.Tensor,
        bias_shift: Union[float, torch.Tensor] = 0.0,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        Q = self.q_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if isinstance(bias_shift, torch.Tensor) or bias_shift != 0.0:
            scores = scores + bias_shift

        attn_weights = torch.softmax(scores, dim=-1)
        context = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(context)

    # ------------------------------------------------------------------
    # Internal: compute gating signals
    # ------------------------------------------------------------------

    def _compute_gates(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (gate_attention, gate_ff), each of shape (B, T, D)."""
        return self.attn_gate_net(x), self.ff_gate_net(x)


    def forward_pass(
        self, x: torch.Tensor, valence: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        gate_attn, gate_ff = self._compute_gates(x)
        bias_shift: Union[float, torch.Tensor] = 0.0
        if valence is not None:
            # Project scalar valence → per-head bias so different heads are
            # shifted by different amounts. Shape: (B, num_heads, 1, 1)
            # broadcasts over the (T_q, T_k) attention score matrix.
            #
            # Bug fix: the old code multiplied valence by a single scalar
            # emotional_gate and added the result uniformly to all logits.
            # Softmax is shift-invariant — a constant added to every logit
            # produces identical attention weights, so the gate did nothing.
            v = valence.reshape(B, 1) if valence.dim() >= 1 else valence.expand(B, 1)
            head_bias = self.emotion_head_router(v)          # (B, num_heads)
            bias_shift = head_bias.view(B, self.num_heads, 1, 1)  # broadcast T×T
        attn_out = self._multi_head_attention(x, bias_shift=bias_shift)
        x = self.norm1(x + gate_attn * attn_out)
        # Feed-forward network step
        ff_out = self.ff_net(x)
        x = self.norm2(x + gate_ff * ff_out)
        return x
