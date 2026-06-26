from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from core.latent_alignment import LatentAligner

class MoodState:
    def __init__(self, initial_mood: str, valid_vocab: dict) -> None:
        if initial_mood not in valid_vocab:
            raise ValueError(
                f"Initial mood '{initial_mood}' not in vocabulary: "
                f"{list(valid_vocab.keys())}"
            )
        self._mood: str = initial_mood
        self._reason: str = "Initialization"
        self._vocab: dict = valid_vocab

    @property
    def name(self) -> str:
        return self._mood

    @property
    def reason(self) -> str:
        return self._reason

    def transition(self, new_mood: str, reason: str) -> None:
        if new_mood not in self._vocab:
            print(f"[MoodState] Unknown mood '{new_mood}'; transition ignored.")
            return
        self._mood = new_mood
        self._reason = reason

class HomeostasisDynamicsNet(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 16, output_dim: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()  # outputs changes scaled between -1 and 1
        )

    def forward(self, state: torch.Tensor, action_impact: float, environment_surprise: float) -> torch.Tensor:
        device = state.device
        dtype = state.dtype
        if state.dim() == 1:
            impact_t = torch.tensor([action_impact], device=device, dtype=dtype)
            surprise_t = torch.tensor([environment_surprise], device=device, dtype=dtype)
            inputs = torch.cat([state, impact_t, surprise_t], dim=0)
            return self.net(inputs)
        else:
            B = state.size(0)
            impact_t = torch.full((B, 1), action_impact, device=device, dtype=dtype)
            surprise_t = torch.full((B, 1), environment_surprise, device=device, dtype=dtype)
            inputs = torch.cat([state, impact_t, surprise_t], dim=-1)
            return self.net(inputs)


class HomeostasisState(nn.Module):

    def __init__(
        self,
        initial: Tuple[float, float, float, float] = (0.5, 0.8, 1.0, 0.5),
        target: Tuple[float, float, float, float] = (0.8, 1.0, 0.7, 0.6),
    ) -> None:
        super().__init__()
        self._state = nn.Parameter(torch.tensor(list(initial)), requires_grad=False)
        self._target = torch.tensor(list(target))
        self.dynamics_net = HomeostasisDynamicsNet()

    @property
    def vector(self) -> torch.Tensor:
        return self._state

    def update(self, action_impact: float, environment_surprise: float) -> None:
        with torch.no_grad():
            state_detached = self._state.detach()
            delta = self.dynamics_net(state_detached, action_impact, environment_surprise)
            new_state = torch.clamp(self._state + 0.1 * delta, 0.0, 1.0)
            self._state.copy_(new_state)

    def strain(self) -> torch.Tensor:
        target = self._target.to(self._state.device)
        return torch.norm(self._state - target)


class EmotionalCore(nn.Module):

    def __init__(
        self,
        latent_aligner: LatentAligner,
        initial_mood: str = "Calm",
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self._aligner = latent_aligner

        self._mood = MoodState(initial_mood, latent_aligner.emotion_vocab)
        self._homeostasis = HomeostasisState()
        self.internal_state = self._homeostasis._state
        self.valence_network = nn.Sequential(
            nn.Linear(hidden_dim + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )


    def current_mood(self) -> Tuple[str, torch.Tensor]:
        mood_id = self._aligner.emotion_vocab[self._mood.name]
        idx = torch.tensor(
            [mood_id],
            dtype=torch.long,
            device=self._aligner.emotion_embeddings.weight.device,
        )
        vector = self._aligner.emotion_embeddings(idx).squeeze(0).detach()
        return self._mood.name, vector

    def _mood_swing(self, new_mood: str, reason: str) -> None:
        self._mood.transition(new_mood, reason)

    def reason_for_mood_change(self) -> str:
        return self._mood.reason

    def set_mood_angry(self, reason: str) -> None:
        self._mood_swing("Angry", reason)

    def set_mood_sad(self, reason: str) -> None:
        self._mood_swing("Sad", reason)

    def set_mood_happy(self, reason: str) -> None:
        self._mood_swing("Happy", reason)

    def set_mood_calm(self, reason: str) -> None:
        self._mood_swing("Calm", reason)

    def update_homeostasis(self, action_impact: float, environment_surprise: float) -> None:
        self._homeostasis.update(action_impact, environment_surprise)

    def calculate_strain(self) -> torch.Tensor:
        return self._homeostasis.strain()

    def get_valence(self, latent_state: torch.Tensor) -> torch.Tensor:
        s = self._homeostasis.vector
        if latent_state.dim() > 1:
            s = s.unsqueeze(0).expand(latent_state.size(0), -1)
        combined = torch.cat([latent_state, s], dim=-1)
        return self.valence_network(combined)
