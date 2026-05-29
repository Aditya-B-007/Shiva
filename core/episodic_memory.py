from __future__ import annotations
import random
from collections import deque
import torch
import torch.nn as nn
from core.interfaces import IEpisodicMemory

class EpisodicMemory(IEpisodicMemory):
    """
    Significance-weighted episodic replay memory.

    Each episode is stored as a dictionary:
        { 'states': Tensor[T, D],  'significance': float }

    Significance is defined as:
        σ = |mean(valence_sequence)| + empowerment_score

    This prioritises emotionally salient or empowering experiences during
    the dreaming phase, preserving the original mathematical intent.

    The identity context is computed by passing the current latent through
    a GRU and adding a learnable self-token:
        h = GRU(z)[−1]
        identity_context = h + self_token
    """

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

        # Narrative encoder: single-layer GRU over the latent sequence.
        self.narrative_encoder = nn.GRU(latent_dim, latent_dim, batch_first=True)

        # Learnable self-token: distinguishes the agent's own representations
        # from world representations during identity context construction.
        self.self_token = nn.Parameter(torch.randn(1, 1, latent_dim))

    # ------------------------------------------------------------------
    # IEpisodicMemory implementation
    # ------------------------------------------------------------------

    def store_episode(
        self,
        state_sequence: torch.Tensor,
        valence_sequence: torch.Tensor,
        empowerment_score: float,
    ) -> None:
        """
        Compute episode significance and append to the bounded memory bank.

          σ = |E[valence]| + empowerment_score
        """
        significance = torch.abs(valence_sequence.mean()) + empowerment_score
        self._bank.append(
            {
                "states": state_sequence.detach(),
                "significance": significance.item(),
            }
        )

    def get_dream_batch(self, batch_size: int) -> torch.Tensor | None:
        """
        Sample a batch of episodes weighted by their significance scores.
        Returns None if fewer than batch_size episodes have been stored.
        """
        if len(self._bank) < batch_size:
            return None
        weights = [ep["significance"] for ep in self._bank]
        samples = random.choices(list(self._bank), weights=weights, k=batch_size)
        return torch.stack([s["states"] for s in samples])

    def get_identity_context(self, current_latent: torch.Tensor) -> torch.Tensor:
        """
        Produce an identity-grounded context vector.

          h_n = GRU(z.unsqueeze(1))[−1]          # final hidden state
          identity_context = h_n + self_token     # ground in self

        Returns a tensor of shape (D,) matching current_latent.
        """
        _, h_n = self.narrative_encoder(current_latent.unsqueeze(1))
        # h_n: (num_layers=1, batch, latent_dim) → squeeze to (batch, latent_dim)
        identity_context = h_n[-1]
        return identity_context + self.self_token.squeeze(0)
