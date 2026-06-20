from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from core.interfaces import IActor, IEpisodicMemory


# ---------------------------------------------------------------------------
# Continuous actor
# ---------------------------------------------------------------------------

class ContinuousActor(IActor):

    def __init__(self, d_model: int, action_dim: int) -> None:
        super().__init__()
        self.mu = nn.Linear(d_model, action_dim)
        self.log_std = nn.Linear(d_model, action_dim)

    def forward(
        self, state_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_std) — the distribution parameters."""
        mu = self.mu(state_features)
        log_std = torch.clamp(self.log_std(state_features), -20, 2)
        return mu, log_std

    def sample(
        self, state_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        mu, log_std = self.forward(state_features)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Double Q-critic
# ---------------------------------------------------------------------------

class DoubleQCritic(nn.Module):

    def __init__(self, d_model: int, action_dim: int) -> None:
        super().__init__()
        self.critic1 = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.critic2 = nn.Sequential(
            nn.Linear(d_model + action_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self, z_conscious: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([z_conscious, action], dim=-1)
        return self.critic1(sa), self.critic2(sa)


# ---------------------------------------------------------------------------
# Continuous SAC policy
# ---------------------------------------------------------------------------

class ContinuousSACPolicy(nn.Module):

    def __init__(
        self,
        backbone: nn.Module,
        actor1: IActor,
        actor2: IActor,
        memory: IEpisodicMemory,
        critic: DoubleQCritic,
        d_model: int,
        swarm=None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.actor1 = actor1
        self.actor2 = actor2
        self.memory = memory
        self.critic = critic
        self.swarm=swarm
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

    def get_action(self, state: torch.Tensor):
        
        z = self.backbone.forward_pass(state)          # (B, T, D)
        z_global = z.mean(dim=1)                        # (B, D)
        z_id = self.memory.get_identity_context(z_global)
        z_conscious = z_global + z_id
        if self.swarm is not None:
            self.swarm.update_node_latent(
                "local_node",
                z_conscious.mean(dim=0)
            )

            consensus,_=self.swarm.step()

            z_conscious=(
                z_conscious
                +
                consensus.unsqueeze(0)
            )
        g = self.gate(z_conscious)                      # (B, 1)

        mu1, log_std1 = self.actor1.forward(z_conscious)
        mu2, log_std2 = self.actor2.forward(z_conscious)

        blended_mu = g * mu1 + (1 - g) * mu2
        blended_var = g * torch.exp(2 * log_std1) + (1 - g) * torch.exp(2 * log_std2)
        std = torch.sqrt(blended_var + 1e-8)
        dist = Normal(blended_mu, std)
        x_t = dist.rsample()
        final_action = torch.tanh(x_t)
        final_log_prob = dist.log_prob(x_t) - torch.log(
            1 - final_action.pow(2) + 1e-6
        )
        return final_action, final_log_prob.sum(dim=-1, keepdim=True), g

    def evaluate_q(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z_global = self.backbone.forward_pass(state).mean(dim=1)  # (B, D)
        z_id = self.memory.get_identity_context(z_global)
        z_conscious = z_global + z_id
        if self.swarm is not None:
            self.swarm.update_node_latent(
                "local_node",
                z_conscious.mean(dim=0)
            )
            consensus, _ = self.swarm.step()
            z_conscious = z_conscious + consensus.unsqueeze(0)
        return self.critic(z_conscious, action)


# ---------------------------------------------------------------------------
# Discrete valence policy
# ---------------------------------------------------------------------------

class DiscreteValencePolicy(nn.Module):

    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, action_dim),
        )
        self.value_manifold = nn.Linear(state_dim, 1)

    def get_empowerment(self, action_probs: torch.Tensor) -> torch.Tensor:
        # Empowerment = E_s[KL(p(a|s) || p̄(a))] = H(p̄) - E_s[H(p(·|s))]
        # i.e. how much knowing the state reduces action uncertainty on average.
        #
        # Bug fix: the old formula computed log(p/q + ε) — the outer +ε was
        # outside the log, corrupting the KL quantity. Also, 0·log(0) → NaN
        # with naive division. torch.xlogy handles the 0·log(0)=0 convention
        # natively, making this both correct and numerically stable.
        marginal = action_probs.mean(dim=0, keepdim=True)          # (1, A)
        kl = torch.sum(
            torch.xlogy(action_probs, action_probs)                 # p·log(p)
            - torch.xlogy(action_probs, marginal.clamp(min=1e-9)), # p·log(q)
            dim=-1,
        )  # (B,)
        return kl.mean()

    def forward(
        self, state: torch.Tensor, valence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        logits = self.actor(state)
        action_probs = F.softmax(logits + valence, dim=-1)
        empowerment = self.get_empowerment(action_probs)
        return action_probs, empowerment
