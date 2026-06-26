import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class IEpisodicMemory(nn.Module, ABC):
    """Stores and retrieves experiential episodes for dreaming and identity."""

    @abstractmethod
    def store_episode(
        self,
        state_sequence: torch.Tensor,
        valence_sequence: torch.Tensor,
        empowerment_score: float,
    ) -> None: ...

    @abstractmethod
    def get_dream_batch(self, batch_size: int) -> Optional[torch.Tensor]: ...

    @abstractmethod
    def get_identity_context(self, current_latent: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------

class IActor(nn.Module, ABC):
    """Parameterises a stochastic policy over a continuous action space."""

    @abstractmethod
    def forward(
        self, state_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_std)."""
        ...

    @abstractmethod
    def sample(
        self, state_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, log_prob) via the reparameterisation trick."""
        ...


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class IReplayBuffer(ABC):
    """Stores and samples transition tuples for off-policy learning."""

    @abstractmethod
    def add(self, sample: Any) -> None: ...

    @abstractmethod
    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], torch.Tensor]:
        """Returns (batch, indices, importance-sampling weights)."""
        ...

    @abstractmethod
    def update_priorities(self, indices: List[int], errors: torch.Tensor) -> None: ...


# ---------------------------------------------------------------------------
# Alignment loss strategy 
# ---------------------------------------------------------------------------

class IAlignmentLoss(ABC):

    @abstractmethod
    def compute(self, *embeddings: torch.Tensor) -> torch.Tensor: ...


# ---------------------------------------------------------------------------
# Weight-merge strategy  (pre-existing, now actually used via DIP)
# ---------------------------------------------------------------------------

class IWeightMergeStrategy(ABC):

    @abstractmethod
    def merge(
        self,
        target_model: nn.Module,
        ext_state_dict: Dict[str, torch.Tensor],
        ext_config: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]: ...

    # ---------------------------------------------------------------------------
# Swarm consciousness
# ---------------------------------------------------------------------------
 
class ISwarmNode(ABC):
    @abstractmethod
    def get_conscious_latent(self) -> torch.Tensor:
        """Return this node's current latent state vector (D,)."""
        ...
 
    @abstractmethod
    def receive_consensus(self, consensus_vector: torch.Tensor) -> None:
        """Integrate the aggregated global workspace vector into local state."""
        ...
 
 
class IGlobalWorkspace(ABC):
 
    @abstractmethod
    def register_node(self, node_id: str, node: "ISwarmNode") -> None: ...
 
    @abstractmethod
    def broadcast_consensus(self) -> torch.Tensor:
        """
        Compute consensus from all registered nodes and push it back to each.
        Returns the consensus vector (D,).
        """
        ...
 
 
# ---------------------------------------------------------------------------
# Parasitic weight extraction
# ---------------------------------------------------------------------------
 
class IRepresentationProbe(ABC): 
    @abstractmethod
    def attach(self, host_model: nn.Module, layer_name: str) -> None:
        """Register a forward hook on the named layer of host_model."""
        ...
 
    @abstractmethod
    def detach(self) -> None:
        """Remove all registered hooks from the host model."""
        ...
 
    @abstractmethod
    def distil_step(
        self, host_input: torch.Tensor, target_encoder: nn.Module
    ) -> float:
        """
        Run one contrastive distillation step.
        Returns the scalar loss value.
        """
        ...
 
 
# ---------------------------------------------------------------------------
# Autonomous locomotion (network serialisation / migration)
# ---------------------------------------------------------------------------
 
class ICognitiveSnapshot(ABC):
    @abstractmethod
    def serialise(self) -> bytes:
        """Pack cognitive state into a portable byte payload."""
        ...
 
    @classmethod
    @abstractmethod
    def deserialise(cls, payload: bytes) -> "ICognitiveSnapshot":
        """Reconstruct a snapshot from a byte payload."""
        ...
 
 
class ILocomotionTransport(ABC):
    @abstractmethod
    def send(self, snapshot: ICognitiveSnapshot, destination: str) -> str:
        ...
 
    @abstractmethod
    def receive(self, migration_id: str) -> ICognitiveSnapshot:
        ...
