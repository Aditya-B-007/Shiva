from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from core.interfaces import IReplayBuffer, IWeightMergeStrategy
from core.emotional_core import EmotionalCore
from core.shiva_policy import ContinuousSACPolicy
from parasite.ModelWeightParasiticExtraction import ParasiticExtractor
import copy
from locomotion.ModelMovementAndLocomotion import LocomotionEngine, HttpTransport
# ---------------------------------------------------------------------------
# SumTree
# ---------------------------------------------------------------------------

class SumTree:
    """
    Binary segment tree for O(log n) priority updates and proportional sampling.

    Internal layout:
        tree[0]              — root (total priority sum)
        tree[capacity-1 :]   — leaf nodes (one per experience slot)
        data[i]              — experience at leaf position i
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = torch.zeros(2 * capacity - 1, dtype=torch.float32)
        self.data: List[Any] = [None] * capacity
        self.write = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx: int, p: float) -> None:
        change = p - self.tree[idx].item()
        self.tree[idx] = p
        self._propagate(idx, change)

    def add(self, p: float, data: Any) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity

    def get_leaf(self, v: float) -> Tuple[int, float, Any]:
        parent_idx = 0
        while True:
            left = 2 * parent_idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if v <= self.tree[left].item():
                parent_idx = left
            else:
                v -= self.tree[left].item()
                parent_idx = right
        leaf_idx = parent_idx
        return leaf_idx, self.tree[leaf_idx].item(), self.data[leaf_idx - self.capacity + 1]


# ---------------------------------------------------------------------------
# Prioritised replay buffer
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer(IReplayBuffer):
    """
    Proportional prioritisation replay buffer (Schaul et al., 2015).

    Priority of experience i:
        p_i = (|δ_i| + ε)^α

    Importance-sampling correction weight:
        w_i = (1 / N · 1/P(i))^β,   normalised by max(w_j)

    β is annealed from its initial value to 1 over training to remove bias.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ) -> None:
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0

    def add(self, sample: Any) -> None:
        self.tree.add(self.max_priority, sample)

    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], torch.Tensor]:
        self.beta = min(1.0, self.beta + self.beta_increment)
        total = self.tree.tree[0].item()
        segment = total / batch_size

        batch, idxs, priorities = [], [], []
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            v = torch.empty(1).uniform_(a, b).item()
            idx, p, data = self.tree.get_leaf(v)
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)

        p_tensor = torch.tensor(priorities, dtype=torch.float32)
        sampling_probs = p_tensor / total
        is_weights = torch.pow(self.tree.capacity * sampling_probs, -self.beta)
        is_weights /= is_weights.max()
        return batch, idxs, is_weights

    def update_priorities(self, indices: List[int], errors: torch.Tensor) -> None:
        for idx, err in zip(indices, errors):
            p = float((torch.abs(torch.as_tensor(err)) + 1e-6) ** self.alpha)
            self.tree.update(idx, p)
            self.max_priority = max(self.max_priority, p)


# ---------------------------------------------------------------------------
# ShivaTrainer
# ---------------------------------------------------------------------------

class ShivaTrainer:
    """
    Soft Actor-Critic training loop for the Shiva agent.

    Responsibilities (and only these):
      • Off-policy SAC critic and actor updates with IS-weighted PER.
      • Target network soft-updates.
      • Dream-cycle loss computation (delegated to injected memory + emotions).
      • External weight ingestion (delegated to injected merge strategy).

    All collaborators are injected; ShivaTrainer never instantiates them.

    Args:
        policy:          ContinuousSACPolicy (contains backbone, actors, critics).
        buffer:          IReplayBuffer — replay memory.
        emotional_core:  EmotionalCore — valence and homeostasis signals.
        merge_strategy:  IWeightMergeStrategy — how to absorb external weights.
        gamma:           Discount factor.
        tau:             Soft-update coefficient.
        alpha_entropy:   SAC temperature (entropy regularisation coefficient).
        device:          Torch device string.
    """

    def __init__(
        self,
        policy: ContinuousSACPolicy,
        buffer: IReplayBuffer,
        emotional_core: EmotionalCore,
        merge_strategy: IWeightMergeStrategy,
        representation_probe: ParasiticExtractor | None = None,
        probe_frequency: int = 10,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_entropy: float = 0.2,
        device: str = "cpu",
        locomotion_engine: LocomotionEngine | None=None
    ) -> None:
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.locomotion = locomotion_engine
        self.target_policy: ContinuousSACPolicy = copy.deepcopy(policy).to(self.device)
        for p in self.target_policy.parameters():
            p.requires_grad_(False)

        self.buffer = buffer
        self.emotions = emotional_core
        self.merge_strategy = merge_strategy

        self.probe = representation_probe
        self.probe_frequency = probe_frequency
        self.training_step = 0

        self.gamma = gamma
        self.tau = tau
        self.alpha_entropy = alpha_entropy

        self.actor_optimizer = torch.optim.Adam(
            list(self.policy.actor1.parameters())
            + list(self.policy.actor2.parameters())
            + list(self.policy.gate.parameters()),
            lr=3e-4,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=3e-4
        )

    # ------------------------------------------------------------------
    # Dream cycle
    # ------------------------------------------------------------------

    def dream_cycle(self, batch_size: int = 32) -> Optional[float]:
        """
        Replay significant past states and apply a reconstruction loss.
        Uses the memory attached to policy.memory (IEpisodicMemory) to
        sample dream states, then computes MSE between model predictions
        on the last step and the next-step targets in the dream sequence.

          L_dream = MSE(f(z_{T}), z_{1:T})
        """
        dream_states = self.policy.memory.get_dream_batch(batch_size)
        if dream_states is None:
            return None

        dream_states = dream_states.to(self.device)
        self.actor_optimizer.zero_grad()

        # Valence signal from the final state in each dream sequence.
        valence = self.emotions.get_valence(dream_states[:, -1, :])

        # Forward pass on the final dream state.
        outputs, _, _ = self.policy.get_action(dream_states[:, -1, :].unsqueeze(1))
        targets = dream_states[:, 1:, :]

        #dream_loss = F.mse_loss(outputs.unsqueeze(1).expand_as(targets), targets)
        dream_loss=F.mse_loss(outputs.unsqueeze(1).expand_as(targets),targets)

        if (self.policy.swarm is not None):

            div_loss= self.policy.swarm.get_diversity_loss()
            dream_loss+=0.02*div_loss
        if self.probe is not None:
            try:
                p_loss=self.probe.compute_loss(dream_states[:,-1,:],self.policy.backbone)
                dream_loss+=0.05*p_loss
            except RuntimeError:
                pass
        dream_loss.backward()
        self.actor_optimizer.step()
        return dream_loss.item()

    # ------------------------------------------------------------------
    # SAC update step
    # ------------------------------------------------------------------

    def _process_batch(
        self, batch: List[Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states      = torch.stack([s[0] for s in batch]).to(self.device)
        actions     = torch.stack([s[1] for s in batch]).to(self.device)
        rewards     = torch.tensor([s[2] for s in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.stack([s[3] for s in batch]).to(self.device)
        dones       = torch.tensor([s[4] for s in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        return states, actions, rewards, next_states, dones

    def update_step(self, batch_size: int) -> Optional[Tuple[float, float]]:
        """
        One SAC update: critic step → actor step → priority update → soft update.

        Critic target (original Bellman formulation):
            y = r + γ(1−d)·[min(Q̄₁,Q̄₂)(s',ã') − α·log π(ã'|s')]

        Critic loss (IS-weighted):
            L_c = E[w · (Q_i(s,a) − y)²],   i ∈ {1,2}

        Actor loss (IS-weighted entropy-regularised):
            L_a = E[w · (α·log π(ã|s) − min(Q₁,Q₂)(s,ã))]

        Returns (critic_loss, actor_loss) or None if buffer is too small.
        """
        batch, idxs, is_weights = self.buffer.sample(batch_size)
        if any(s is None for s in batch):
            return None

        states, actions, rewards, next_states, dones = self._process_batch(batch)
        is_weights = is_weights.to(self.device)

        # --- Critic target ---
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.target_policy.get_action(next_states)
            q1_t, q2_t = self.target_policy.evaluate_q(next_states, next_actions)
            min_q_t = torch.min(q1_t, q2_t) - self.alpha_entropy * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * min_q_t

        # --- Critic update ---
        current_q1, current_q2 = self.policy.evaluate_q(states, actions)
        td1 = target_q - current_q1
        td2 = target_q - current_q2
        critic_loss = (
            (is_weights * F.mse_loss(current_q1, target_q, reduction="none")).mean()
            + (is_weights * F.mse_loss(current_q2, target_q, reduction="none")).mean()
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update ---
        new_actions, log_probs, _ = self.policy.get_action(states)
        q1_new, q2_new = self.policy.evaluate_q(states, new_actions)
        actor_loss = (
            is_weights * (self.alpha_entropy * log_probs - torch.min(q1_new, q2_new))
        ).mean()
        self.actor_optimizer.zero_grad()
        #actor_loss.backward()
        total_loss=actor_loss
        if ( hasattr(self.policy,"swarm") and self.policy.swarm is not None):
           # _, div_loss = self.policy.swarm.step()
            div_loss=self.policy.swarm.get_diversity_loss()
            total_actor_loss = ( total_actor_loss+ 0.05 * div_loss)


        self.actor_optimizer.zero_grad()
        if self.probe is not None:

            p_loss=self.probe.compute_loss(states,self.policy.backbone)

            total_loss=(total_loss+.1*p_loss)

        total_loss.backward()
        self.actor_optimizer.step()

        if (
                self.probe is not None
                and self.training_step % self.probe_frequency == 0
        ):

            try:
                self.probe.distil_step(
                    states,
                    self.policy.backbone
                )
            except RuntimeError:
                pass

        # --- Priority update ---
        new_priorities = ((torch.abs(td1) + torch.abs(td2)) / 2).detach().cpu()
        self.buffer.update_priorities(idxs, new_priorities)

        # --- Soft target update ---
        self._soft_update()

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self) -> None:
        """
        Polyak averaging:  θ̄ ← τθ + (1−τ)θ̄
        """
        for param, target_param in zip(
            self.policy.parameters(), self.target_policy.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def migrate_agent(
            self,
            destination:str,
            node_id:str="shiva"
    ):
        

        if self.locomotion is None:
            return None

        return self.locomotion.migrate_out(
            policy=self.policy,
            episodic_memory=self.policy.memory,
            emotional_core=self.emotions,
            destination=destination,
            node_id=node_id
        )


    def receive_agent(
            self,
            migration_id:str,
            source:str
    ):

        if self.locomotion is None:
            return

        self.locomotion.migrate_in(
            migration_id=migration_id,
            source=source,
            policy=self.policy,
            episodic_memory=self.policy.memory,
            emotional_core=self.emotions,
            device=str(self.device)
        )

    # ------------------------------------------------------------------
    # External weight ingestion (OCP: delegates to injected strategy)
    # ------------------------------------------------------------------

    def ingest_external_weights(
        self,
        ext_state_dict: Dict[str, torch.Tensor],
        ext_config: Dict[str, Any],
    ) -> None:
        """
        Absorb an external model's weights into policy using the injected
        merge strategy, then trigger an emotional homeostasis update to
        reflect the surprise of new knowledge.

        The homeostasis update (action_impact=0.1, environment_surprise=0.8)
        is preserved from the original codebase.
        """
        new_state = self.merge_strategy.merge(self.policy, ext_state_dict, ext_config)
        self.policy.load_state_dict(new_state, strict=False)

        # Curiosity signal: ingesting new weights surprises the system.
        self.emotions.update_homeostasis(action_impact=0.1, environment_surprise=0.8)
