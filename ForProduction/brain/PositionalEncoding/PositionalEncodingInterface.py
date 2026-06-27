from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, *args, **kwargs):
        pass
