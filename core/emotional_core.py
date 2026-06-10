from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from core.latent_alignment import LatentAligner
# ---------------------------------------------------------------------------
# Discrete mood tracking
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
#  Homeostatic drive vector
# ---------------------------------------------------------------------------

class HomeostasisState:
    """
    Dimensions (by convention, mirroring the original codebase):
      [0] arousal-like  – increases with environmental surprise
      [1] energy-like   – decreases with action cost
      [2] safety        – placeholder (fixed at initialisation)
      [3] engagement    – placeholder (fixed at initialisation)

      update: s[0] += 0.05 * ε_env,  s[1] -= 0.02 * δ_action,  clamp to [0,1]
      strain: ‖s − s*‖₂
    """

    def __init__(
        self,
        initial: Tuple[float, float, float, float] = (0.5, 0.8, 1.0, 0.5),
        target: Tuple[float, float, float, float] = (0.8, 1.0, 0.7, 0.6),
    ) -> None:
        self._state = nn.Parameter(torch.tensor(list(initial)), requires_grad=False)
        self._target = torch.tensor(list(target))

    @property
    def vector(self) -> torch.Tensor:
        return self._state

    def update(self, action_impact: float, environment_surprise: float) -> None:
        self._state[0] += 0.05 * environment_surprise
        self._state[1] -= 0.02 * action_impact
        self._state.clamp_(0, 1)

    def strain(self) -> torch.Tensor:
        target = self._target.to(self._state.device)
        return torch.norm(self._state - target)


# ---------------------------------------------------------------------------
# EmotionalCore: orchestrates mood + homeostasis + valence network
# ---------------------------------------------------------------------------

class EmotionalCore(nn.Module):
    """
    Affective subsystem exposed to the rest of the Shiva architecture.

    Responsibilities (and only these):
      1. Delegate discrete mood bookkeeping to MoodState.
      2. Delegate homeostatic drive updates to HomeostasisState.
      3. Run the learned valence network that maps (latent_state ‖ internal_state)
         → scalar valence ∈ (−1, +1).

    The emotion embedding lookup is forwarded to LatentAligner, keeping a
    single source of truth for the emotion vocabulary and its embeddings.
    """

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

    # ------------------------------------------------------------------
    # Mood interface
    # ------------------------------------------------------------------

    def current_mood(self) -> Tuple[str, torch.Tensor]:
        mood_id = self._aligner.emotion_vocab[self._mood.name]
        idx = torch.tensor(
            [mood_id],
            dtype=torch.long,
            device=self._aligner.emotion_embeddings.weight.device,
        )
        vector = self._aligner.emotion_embeddings(idx).squeeze(0).detach()
        return self._mood.name, vector

    def mood_swing(self, new_mood: str, reason: str) -> None:
        self._mood.transition(new_mood, reason)

    def reason_for_mood_change(self) -> str:
        return self._mood.reason

    def set_mood_angry(self, reason: str) -> None:
        self.mood_swing("Angry", reason)

    def set_mood_sad(self, reason: str) -> None:
        self.mood_swing("Sad", reason)

    def set_mood_happy(self, reason: str) -> None:
        self.mood_swing("Happy", reason)

    def set_mood_calm(self, reason: str) -> None:
        self.mood_swing("Calm", reason)

    # ------------------------------------------------------------------
    # Homeostasis interface
    # ------------------------------------------------------------------

    def update_homeostasis(self, action_impact: float, environment_surprise: float) -> None:
        self._homeostasis.update(action_impact, environment_surprise)

    def calculate_strain(self) -> torch.Tensor:
        return self._homeostasis.strain()

    # ------------------------------------------------------------------
    # Valence network
    # ------------------------------------------------------------------

    def get_valence(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Compute affective valence for a given latent state.

          v = tanh(W₂ · ReLU(W₁ · [z ‖ s]) )

        Handles both batched (B, D) and unbatched (D,) inputs.
        """
        s = self._homeostasis.vector
        if latent_state.dim() > 1:
            s = s.unsqueeze(0).expand(latent_state.size(0), -1)
        combined = torch.cat([latent_state, s], dim=-1)
        return self.valence_network(combined)
