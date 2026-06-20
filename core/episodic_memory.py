from __future__ import annotations
import random
from collections import deque
import torch
import torch.nn as nn
from core.interfaces import IEpisodicMemory

class EpisodicMemory(IEpisodicMemory):

    def __init__(
        self,
        latent_dim: int = 512,
        capacity: int = 10000,
        sequence_length: int = 16,
    ) -> None:
        super().__init__()
        self.capacity = capacity
        self.sequence_length = sequence_length

        self._bank: deque = deque(maxlen=capacity)
        self.narrative_encoder = nn.GRU(latent_dim, latent_dim, batch_first=True,num_layers=1)
        self.self_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    def store_episode(
        self,
        state_sequence: torch.Tensor,
        valence_sequence: torch.Tensor,
        empowerment_score: float,
    ) -> None:
        significance = torch.abs(valence_sequence.mean()) + empowerment_score
        self._bank.append(
            {
                "states": state_sequence.detach(),
                "significance": significance.item(),
            }
        )

    def get_dream_batch(self, batch_size: int) -> torch.Tensor | None:
        if len(self._bank) < batch_size:
            return None
        weights = [ep["significance"] for ep in self._bank]
        samples = random.choices(list(self._bank), weights=weights, k=batch_size)
        return torch.stack([s["states"] for s in samples])


    def get_identity_context(self, current_latent: torch.Tensor) -> torch.Tensor:
        if current_latent.dim() == 1:
            x = current_latent.unsqueeze(0).unsqueeze(1)
            is_batched = False
        else:
            x = current_latent.unsqueeze(1)
            is_batched = True
        
        # Expand self_token along the batch dimension to initialize the GRU hidden state (h_0)
        B = x.size(0)
        h_0 = self.self_token.expand(1, B, -1).contiguous()
        
        self.narrative_encoder.flatten_parameters()
        _, h_n = self.narrative_encoder(x, h_0)
        identity_context = h_n[-1]
        
        # Add self_token to identity_context to preserve learnable self context identity
        combined = identity_context + self.self_token.squeeze(0)
        
        if not is_batched:
            return combined.squeeze(0)
        return combined
