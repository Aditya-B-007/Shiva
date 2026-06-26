from __future__ import annotations
from typing import Any, Dict
import torch
import torch.linalg as linalg
import torch.nn as nn
import torch.nn.functional as F
from core.interfaces import IWeightMergeStrategy

class SVDDimensionFitter:

    @staticmethod
    def fit(W: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        if W.shape == target_shape:
            return W

        # Handle non-2D tensors (e.g. 1D biases or layernorm weights) to prevent crash
        if W.dim() != 2:
            result = torch.zeros(target_shape, dtype=W.dtype, device=W.device)
            slices = tuple(slice(0, min(s_s, t_s)) for s_s, t_s in zip(W.shape, target_shape))
            result[slices] = W[slices]
            return result

        U, S, Vh = linalg.svd(W, full_matrices=False)
        V = Vh.transpose(-2, -1)

        h0, h1 = W.shape
        r0, r1 = target_shape

        k0 = min(h0, r0, U.shape[1])
        P_L = torch.zeros((h0, r0), dtype=W.dtype, device=W.device)
        P_L[:, :k0] = U[:, :k0]

        k1 = min(h1, r1, V.shape[1])
        P_R = torch.zeros((h1, r1), dtype=W.dtype, device=W.device)
        P_R[:, :k1] = V[:, :k1]
        W_projected = P_L.transpose(-2, -1) @ W @ P_R
        return W_projected


class AttentionHeadAverager:

    @staticmethod
    def average(
        W_mha: torch.Tensor,
        src_heads: int,
        target_heads: int,
    ) -> torch.Tensor:
        src_out_dim, d_model = W_mha.shape
        d_head_src = src_out_dim // src_heads
        d_head_target = src_out_dim // target_heads
        x = W_mha.view(1, src_heads, d_head_src, d_model)
        x_interpolated = F.interpolate(
            x,
            size=(target_heads, d_head_target),
            mode="bilinear",
            align_corners=False
        )
        out = x_interpolated.view(target_heads * d_head_target, d_model)
        return out


class RapidFrankenmergeStrategy(IWeightMergeStrategy):

    def __init__(self) -> None:
        self._fitter = SVDDimensionFitter()
        self._head_averager = AttentionHeadAverager()

    def merge(
        self,
        target_model: nn.Module,
        ext_state_dict: Dict[str, torch.Tensor],
        ext_config: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        target_dict = target_model.state_dict()

        if hasattr(target_model, "config"):
            target_heads = target_model.config.num_heads  # type: ignore[union-attr]
        else:
            target_heads = target_model.backbone.num_heads  # type: ignore[union-attr]

        new_state: Dict[str, torch.Tensor] = {}

        src_heads: int = ext_config["num_heads"]

        for name, param in ext_state_dict.items():
            target_shape = (
                target_dict[name].shape
                if name in target_dict
                else param.shape
            )
            is_attention_proj = ("atta" in name or "attention" in name) and param.dim() == 2
            heads_differ = src_heads != target_heads

            if is_attention_proj and heads_differ:
                new_state[name] = self._head_averager.average(
                    param, src_heads, target_heads
                )
            else:
                new_state[name] = self._fitter.fit(param, target_shape)

        return new_state
