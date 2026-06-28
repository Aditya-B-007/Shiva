import torch
from .PositionalEncodingInterface import PositionalEncodingInterface

class ALiBiPositionalEncoding(PositionalEncodingInterface):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        slopes = self._generate_slopes(self.num_heads)
        self.register_buffer("slopes", slopes, persistent=False)

    @staticmethod
    def _generate_slopes(num_heads):
        return torch.tensor([2 ** (-(8 * (i + 1) / num_heads)) for i in range(num_heads)])

    def apply_attention_scores(self, attention_scores: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, _ = attention_scores.shape
        positions = torch.arange(seq, device=attention_scores.device)
        relative = (positions[None, :] - positions[:, None]).abs()
        bias = self.slopes.view(1, heads, 1, 1) * relative
        return attention_scores - bias
