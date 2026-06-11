from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from core.interfaces import IGlobalWorkspace, ISwarmNode
from core.shiva_policy import ContinuousSACPolicy
from core.emotional_core import EmotionalCore
from core.episodic_memory import EpisodicMemory

# ---------------------------------------------------------------------------
# SwarmNode: Full Standalone Cognitive Agent with Shared Heavy Tensors
# ---------------------------------------------------------------------------

class SwarmNode(ISwarmNode, nn.Module):
    def __init__(
        self, 
        node_id: str, 
        latent_dim: int, 
        archetype_policy: ContinuousSACPolicy, 
        archetype_emotional_core: EmotionalCore
    ) -> None:
        nn.Module.__init__(self)
        self.node_id = node_id
        self.latent_dim = latent_dim
        
        # Learnable local integration gate parameter
        self._integration_gate = nn.Parameter(torch.tensor(math.log(0.1 / 0.9)))

        # Persistent unique local conscious latent buffer
        self.register_buffer("_local_latent", torch.zeros(latent_dim))

        # 1. Standalone Unique Memory Structures (Isolated Experience Histories)
        self.memory = EpisodicMemory(latent_dim=latent_dim)
        # Performance Sync: Point narrative weights directly to the parent archetype
        self.memory.narrative_encoder = archetype_policy.memory.narrative_encoder
        self.memory.self_token = archetype_policy.memory.self_token
        
        # 2. Standalone Unique Affective Subsystems (Isolated Emotional Tracks)
        self.emotional_core = EmotionalCore(
            latent_aligner=archetype_emotional_core._aligner, 
            hidden_dim=latent_dim
        )
        # Performance Sync: Point the internal valence MLP to the parent network
        self.emotional_core.valence_network = archetype_emotional_core.valence_network

        # 3. Direct References to Shared Strategic Policy Layers
        self.backbone = archetype_policy.backbone
        self.actor1 = archetype_policy.actor1
        self.actor2 = archetype_policy.actor2
        self.critic = archetype_policy.critic
        self.gate = archetype_policy.gate

    def forward_node(self, local_observation: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Flux internal homeostatic states independently to capture processing friction
        self.emotional_core.update_homeostasis(action_impact=0.04, environment_surprise=0.08)
        
        # Run token streams through the shared multi-head transformer block
        z = self.backbone.forward_pass(local_observation, device)  # Shape: (1, T, D)
        z_global = z.mean(dim=1)  # Shape: (1, D)
        
        # Ground context inside the local separate episodic narrative stack
        z_id = self.memory.get_identity_context(z_global)
        z_conscious = z_global + z_id
        
        # Cache unique conscious trace vector
        self._local_latent = z_conscious.detach().squeeze(0)
        
        # Evaluate expert strategy distributions using shared policy gates
        g = self.gate(z_conscious)
        mu1, log_std1 = self.actor1.forward(z_conscious)
        mu2, log_std2 = self.actor2.forward(z_conscious)
        
        blended_mu = g * mu1 + (1 - g) * mu2
        blended_log_std = g * log_std1 + (1 - g) * log_std2
        
        std = torch.exp(blended_log_std)
        dist = Normal(blended_mu, std)
        x_t = dist.rsample()
        
        final_action = torch.tanh(x_t)
        final_log_prob = dist.log_prob(x_t) - torch.log(1 - final_action.pow(2) + 1e-6)
        
        return final_action, final_log_prob.sum(dim=-1, keepdim=True), g

    def get_conscious_latent(self) -> torch.Tensor:
        return self._local_latent  # type: ignore[return-value]

    def set_conscious_latent(self, z: torch.Tensor) -> None:
        """Manually forces a conscious state write sequence."""
        self._local_latent = z.detach()

    def receive_consensus(self, consensus_vector: torch.Tensor) -> None:
        gate = torch.sigmoid(self._integration_gate)
        self._local_latent = self._local_latent + gate * consensus_vector.detach()
        with torch.no_grad():
            v = self.emotional_core.get_valence(self._local_latent.unsqueeze(0))
            if v.mean().item() > 0.0:
                self.emotional_core.mood_swing("Calm", "Consensus correlates with local discovery streams.")
            else:
                self.emotional_core.mood_swing("Alert", "High variance detected against collective workspace vector.")

# ---------------------------------------------------------------------------
# CrossAttentionAggregator: Baars Global Workspace Conscience Compilation
# ---------------------------------------------------------------------------

class CrossAttentionAggregator(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.d_k = latent_dim // num_heads

        self.W_K = nn.Linear(latent_dim, latent_dim, bias=False)
        self.W_V = nn.Linear(latent_dim, latent_dim, bias=False)
        self.query = nn.Parameter(torch.randn(1, latent_dim))
        self.W_out = nn.Linear(latent_dim, latent_dim)
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

        residual = node_latents.mean(dim=0)
        consensus = self.norm(self.W_out(out) + residual)
        return consensus

# ---------------------------------------------------------------------------
# GlobalWorkspace: Central Anti-Collapse Coordination Platform
# ---------------------------------------------------------------------------

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

        diff = z_norm.unsqueeze(0) - z_norm.unsqueeze(1)  # (N, N, D)
        dist = torch.norm(diff, p=2, dim=-1)               # (N, N)

        mask = torch.triu(torch.ones(N, N, device=latents.device), diagonal=1)
        n_pairs = mask.sum()
        mean_dist = (dist * mask).sum() / n_pairs
        return -mean_dist   # Minimize negative distance = Maximise variance

    @property
    def last_diversity_loss(self) -> Optional[torch.Tensor]:
        return self._last_diversity_loss

# ---------------------------------------------------------------------------
# SwarmCoordinator: Absolute Multi-Node Global System Orchestrator
# ---------------------------------------------------------------------------

class SwarmCoordinator(nn.Module):
    """
    Top-level orchestrator for the decentralized Shiva Swarm.
    Defaults to 100 nodes per call session and scales gracefully to 1,000+ nodes.
    """
    def __init__(
        self,
        latent_dim: int,
        n_nodes: int = 100,  # 100 nodes default configuration
        num_heads: int = 8,
        node_ids: Optional[list] = None,
        archetype_policy: Optional[ContinuousSACPolicy] = None,
        archetype_emotional_core: Optional[EmotionalCore] = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes

        # Runtime safety check: Resolve archetype anchors from global instances if missing
        if archetype_policy is None:
            from managers.runtime import runtime_manager
            archetype_policy = runtime_manager.policy
            archetype_emotional_core = runtime_manager.emotional_core

        ids = node_ids or [f"node_{i}" for i in range(n_nodes)]
        assert len(ids) == n_nodes

        self.workspace = GlobalWorkspace(latent_dim, num_heads)

        # Register nodes inside an optimized ModuleDict container
        self._node_modules = nn.ModuleDict({
            nid: SwarmNode(nid, latent_dim, archetype_policy, archetype_emotional_core) 
            for nid in ids
        })
        for nid, node in self._node_modules.items():
            self.workspace.register_node(nid, node)  # type: ignore[arg-type]

    def update_node_latent(self, node_id: str, z: torch.Tensor) -> None:
        """Writes real-time structural data modifications directly to the node."""
        self._node_modules[node_id].set_conscious_latent(z)

    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Drives a unified consensus cycle across all active nodes."""
        consensus = self.workspace.broadcast_consensus()
        div_loss = self.workspace.last_diversity_loss
        return consensus, div_loss if div_loss is not None else torch.tensor(0.0)

    def get_diversity_loss(self) -> torch.Tensor:
        """Exposes anti-pooling regularization metrics to the centralized trainer optimization paths."""
        div = self.workspace.last_diversity_loss
        return div if div is not None else torch.tensor(0.0)

    def get_node_latent(self, node_id: str) -> torch.Tensor:
        """Returns the specific conscious latent associated with a targeting node id."""
        return self._node_modules[node_id].get_conscious_latent()

    def execute_swarm_step(
        self, 
        partition_inputs: Dict[str, str], 
        device: torch.device
    ) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
        swarm_outputs = {}
        for nid, directive_str in partition_inputs.items():
            if nid in self._node_modules:
                # Triggers standalone execution mapped through shared model references
                swarm_outputs[nid] = self._node_modules[nid].forward_node(directive_str, device)
        
        consensus_vector, diversity_loss = self.step()
        return swarm_outputs, consensus_vector, diversity_loss
