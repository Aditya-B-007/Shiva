from abc import ABC
import torch
import torch.nn as nn

class PositionalEncodingInterface(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def apply_embedding(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def apply_qk(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return q, k

    def apply_attention_scores(self, scores: torch.Tensor) -> torch.Tensor:
        return scores
