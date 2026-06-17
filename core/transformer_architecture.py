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

        # --- Emotional modulation scalar ---
        self.emotional_gate = nn.Parameter(torch.ones(1))

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
        gate_attn, gate_ff = self._compute_gates(x)
        bias_shift: Union[float, torch.Tensor] = 0.0
        if valence is not None:
            raw_bias = valence * self.emotional_gate
            if isinstance(raw_bias, torch.Tensor):
                if raw_bias.dim() == 0: # Pure scalar tensor
                    bias_shift = raw_bias
                else: 
                    bias_shift = raw_bias.view(-1, 1, 1, 1)
            else:
                bias_shift = raw_bias
        attn_out = self._multi_head_attention(x, bias_shift=bias_shift)
        x = self.norm1(x + gate_attn * attn_out)
        # Feed-forward network step
        ff_out = self.ff_net(x)
        x = self.norm2(x + gate_ff * ff_out)
        
        return x
