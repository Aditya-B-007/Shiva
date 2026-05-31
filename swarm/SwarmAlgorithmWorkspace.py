from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from core.interfaces import IGlobalWorkspace, ISwarmNode


# ---------------------------------------------------------------------------
# SwarmNode: one autonomous agent in the collective
# ---------------------------------------------------------------------------

class SwarmNode(ISwarmNode, nn.Module):
    def __init__(self, latent_dim: int, node_id: str) -> None:
        nn.Module.__init__(self)
        self.node_id = node_id
        self.latent_dim = latent_dim
        self._integration_gate = nn.Parameter(
            torch.tensor(math.log(0.1 / 0.9))
        )

        # Persistent local latent (not a parameter — updated each forward).
        self.register_buffer(
            "_local_latent", torch.zeros(latent_dim)
        )

    def set_conscious_latent(self, z: torch.Tensor) -> None:
        """Write the latest latent from the agent's forward pass."""
        self._local_latent = z.detach()

    def get_conscious_latent(self) -> torch.Tensor:
        return self._local_latent  # type: ignore[return-value]

    def receive_consensus(self, consensus_vector: torch.Tensor) -> None:
        gate = torch.sigmoid(self._integration_gate)
        self._local_latent = self._local_latent + gate * consensus_vector.detach()


class CrossAttentionAggregator(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.d_k = latent_dim // num_heads

        # Projections for K and V (applied to the node latent stack).
        self.W_K = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_V = nn.Linear(latent_dim, latent_dim, bias=False)

        # Learnable global query (shared across all aggregation calls).
        self.query = nn.Parameter(torch.randn(1, latent_dim))

        # Output projection.
        self.W_out = nn.Linear(latent_dim, latent_dim)

        # Layer norm for consensus stability.
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, node_latents: torch.Tensor) -> torch.Tensor:
        N, D = node_latents.shape
        H, d_k = self.num_heads, self.d_k

        K = self.W_K(node_latents).view(N, H, d_k).transpose(0, 1)   # (H, N, d_k)
        V = self.W_V(node_latents).view(N, H, d_k).transpose(0, 1)   # (H, N, d_k)

        q = self.query.view(1, H, d_k).transpose(0, 1)                # (H, 1, d_k)

        scores = (q @ K.transpose(-2, -1)) / math.sqrt(d_k)           # (H, 1, N)
        weights = F.softmax(scores, dim=-1)                            # (H, 1, N)

        out = (weights @ V)                                            # (H, 1, d_k)
        out = out.transpose(0, 1).contiguous().view(1, D).squeeze(0)  # (D,)

        # Residual connection with the mean of all node latents.
        residual = node_latents.mean(dim=0)
        consensus = self.norm(self.W_out(out) + residual)
        return consensus

class GlobalWorkspace(IGlobalWorkspace, nn.Module):
    def __init__(self, latent_dim: int, num_heads: int = 8) -> None:
        nn.Module.__init__(self)
        self.latent_dim = latent_dim
        self._nodes: Dict[str, ISwarmNode] = {}
        self.aggregator = CrossAttentionAggregator(latent_dim, num_heads)
        self._last_diversity_loss: Optional[torch.Tensor] = None

    def register_node(self, node_id: str, node: ISwarmNode) -> None:
        self._nodes[node_id] = node

    def broadcast_consensus(self) -> torch.Tensor:
        if not self._nodes:
            raise RuntimeError("No nodes registered in GlobalWorkspace.")
        node_ids = list(self._nodes.keys())
        latents = torch.stack([
            self._nodes[nid].get_conscious_latent() for nid in node_ids
        ])
        consensus = self.aggregator(latents)
        for nid in node_ids:
            self._nodes[nid].receive_consensus(consensus)
        self._last_diversity_loss = self._compute_diversity_loss(latents)
        return consensus

    @staticmethod
    def _compute_diversity_loss(latents: torch.Tensor) -> torch.Tensor:
        z_norm = F.normalize(latents, p=2, dim=1)     # (N, D)
        N = z_norm.shape[0]
        if N < 2:
            return torch.tensor(0.0, device=latents.device)

        # Pairwise L2 via broadcasting: (N, N) matrix of distances.
        diff = z_norm.unsqueeze(0) - z_norm.unsqueeze(1)  # (N, N, D)
        dist = torch.norm(diff, p=2, dim=-1)               # (N, N)

        # Upper triangle only (exclude diagonal and duplicate pairs).
        mask = torch.triu(torch.ones(N, N, device=latents.device), diagonal=1)
        n_pairs = mask.sum()
        mean_dist = (dist * mask).sum() / n_pairs
        return -mean_dist   # negate: maximise distance = minimise this loss

    @property
    def last_diversity_loss(self) -> Optional[torch.Tensor]:
        return self._last_diversity_loss

class SwarmCoordinator(nn.Module):
    """
    Top-level orchestrator for the swarm.

    Responsibilities (SRP):
      • Manage the set of SwarmNodes and the GlobalWorkspace.
      • Drive per-step consensus cycles.
      • Expose the diversity loss for inclusion in the trainer's loss.
      • Provide a consensus-conditioned action for collective decisions.

    Usage pattern:
        coordinator = SwarmCoordinator(latent_dim=512, n_nodes=4)

        # Each agent updates its node with its local latent:
        for i, agent in enumerate(agents):
            z_i = agent.backbone.forward_pass(obs_i).mean(dim=1)
            coordinator.update_node_latent(f"node_{i}", z_i)

        # Run consensus broadcast:
        consensus, diversity_loss = coordinator.step()

        # Each agent's node now has the integrated consensus in its local state.
    """

    def __init__(
        self,
        latent_dim: int,
        n_nodes: int,
        num_heads: int = 8,
        node_ids: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        ids = node_ids or [f"node_{i}" for i in range(n_nodes)]
        assert len(ids) == n_nodes

        self.workspace = GlobalWorkspace(latent_dim, num_heads)

        # Register nodes as submodules so their parameters are tracked.
        self._node_modules = nn.ModuleDict({
            nid: SwarmNode(latent_dim, nid) for nid in ids
        })
        for nid, node in self._node_modules.items():
            self.workspace.register_node(nid, node)  # type: ignore[arg-type]

    def update_node_latent(self, node_id: str, z: torch.Tensor) -> None:
        self._node_modules[node_id].set_conscious_latent(z)  # type: ignore[attr-defined]

    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        consensus = self.workspace.broadcast_consensus()
        div_loss = self.workspace.last_diversity_loss
        return consensus, div_loss if div_loss is not None else torch.tensor(0.0)

    def get_diversity_loss(self):

        div=self.workspace.last_diversity_loss

        return (div if div is not None else torch.tensor(0.0))

    def get_node_latent(self, node_id: str) -> torch.Tensor:
        return self._node_modules[node_id].get_conscious_latent()  # type: ignore[attr-defined]
