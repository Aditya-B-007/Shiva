from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Dict, Any, Optional

class SACActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples action using reparameterization trick and returns action and log prob."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t)
        
        # Calculate log probability with tanh correction
        log_prob = normal.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)


class SwarmSAC:
    def __init__(
        self,
        state_dim: int = 4096,   # Concat of 2048-dim state + 2048-dim goal
        action_dim: int = 2048,  # Action representation vector dimension
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,      # Temperature parameter (entropy scale)
        device: str = "cpu"
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.actor = SACActor(state_dim, action_dim).to(self.device)
        self.critic = SACCritic(state_dim, action_dim).to(self.device)
        self.critic_target = SACCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Simple local replay buffer (state, action, reward, next_state, done)
        self.buffer: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []
        self.max_buffer_size = 1000

    def add_experience(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        if len(self.buffer) >= self.max_buffer_size:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def update_parameters(self, batch_size: int = 32) -> Optional[Tuple[float, float]]:
        """Executes a single step of SAC parameter update."""
        if len(self.buffer) < batch_size:
            return None

        # Sample batch
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        
        states = torch.FloatTensor(np.array([b[0] for b in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([b[1] for b in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).unsqueeze(-1).to(self.device)
        next_states = torch.FloatTensor(np.array([b[3] for b in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([b[4] for b in batch])).unsqueeze(-1).to(self.device)

        # ----------------------------------------------------
        # 1. Update Critic Twin Networks
        # ----------------------------------------------------
        with torch.no_grad():
            next_state_actions, next_state_log_probs = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_state_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_state_log_probs
            y = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------------------------------------
        # 2. Update Actor Network
        # ----------------------------------------------------
        sampled_actions, log_probs = self.actor.sample(states)
        q1, q2 = self.critic(states, sampled_actions)
        q = torch.min(q1, q2)
        actor_loss = (self.alpha * log_probs - q).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------------------------------------
        # 3. Soft Update Target Critic (Polyak Averaging)
        # ----------------------------------------------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()
