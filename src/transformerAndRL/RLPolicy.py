import numpy as np
from typing import List, Optional, Dict, Tuple
try:
    from src.config.dtos import DecisionOutputDTO, ScoredActionDTO
except ImportError:
    from Shiva.src.config.dtos import DecisionOutputDTO, ScoredActionDTO


class MCTSNode:
    def __init__(self, action_idx: int, prior: float):
        self.action_idx: int = action_idx
        self.prior: float = prior                      # Policy prior P(s, a)
        self.visit_count: int = 0                      # N(s, a)
        self.total_value: float = 0.0                  # W(s, a)
        self.mean_value: float = 0.0                   # Q(s, a)


class MCTSSearch:
    def __init__(
        self,
        num_simulations: int = 64,
        c_puct: float = 1.414,
        dirichlet_alpha: float = 0.3,
        dirichlet_weight: float = 0.25
    ):
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_weight = dirichlet_weight

    def search(
        self,
        situation_vector: np.ndarray,
        options_matrix: np.ndarray,
        base_probs: np.ndarray
    ) -> np.ndarray:
        K = options_matrix.shape[0]
        priors = base_probs[0].copy()
        if K > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * K)
            priors = (1.0 - self.dirichlet_weight) * priors + self.dirichlet_weight * noise
        children: Dict[int, MCTSNode] = {
            a: MCTSNode(action_idx=a, prior=float(priors[a])) for a in range(K)
        }
        alignments = np.matmul(situation_vector, options_matrix.T)[0]  # Shape: [K]

        for _ in range(self.num_simulations):
            total_parent_visits = sum(c.visit_count for c in children.values())
            sqrt_parent = np.sqrt(max(total_parent_visits, 1))

            # 1. Selection via PUCT: Q(s, a) + U(s, a)
            best_score = -float("inf")
            best_action = 0

            for a, child in children.items():
                u_score = self.c_puct * child.prior * (sqrt_parent / (1 + child.visit_count))
                puct_score = child.mean_value + u_score

                if puct_score > best_score:
                    best_score = puct_score
                    best_action = a

            selected_child = children[best_action]
            val = float(alignments[best_action])

            # 3. Backpropagation
            selected_child.visit_count += 1
            selected_child.total_value += val
            selected_child.mean_value = selected_child.total_value / selected_child.visit_count

        # Compute search policy from visit counts: N(s, a) / sum(N)
        visit_counts = np.array([children[a].visit_count for a in range(K)], dtype=np.float32)
        mcts_probs = visit_counts / np.sum(visit_counts)

        return mcts_probs[np.newaxis, :]  # Shape: [1, K]


class RLPolicy:
    def __init__(
        self,
        temperature: float = 0.07,
        num_simulations: int = 64,
        c_puct: float = 1.414
    ):
        self.temperature = temperature
        self.mcts = MCTSSearch(num_simulations=num_simulations, c_puct=c_puct)

    def evaluate(
        self,
        situation_vector: np.ndarray,
        options_matrix: np.ndarray,
        candidate_strings: Optional[List[str]] = None,
        enable_thinking: bool = False
    ) -> DecisionOutputDTO:
        K = options_matrix.shape[0]
        if K == 0:
            raise ValueError("Candidate options matrix is empty (K=0).")
        eps = 1e-12
        z_norm = situation_vector / (np.linalg.norm(situation_vector, ord=2, axis=-1, keepdims=True) + eps)
        m_norm = options_matrix / (np.linalg.norm(options_matrix, ord=2, axis=-1, keepdims=True) + eps)

        logits = np.matmul(z_norm, m_norm.T) / self.temperature

        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        base_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # [1, K]

        if enable_thinking:
            final_probs = self.mcts.search(
                situation_vector=z_norm,
                options_matrix=m_norm,
                base_probs=base_probs
            )
        else:
            final_probs = base_probs

        best_idx = int(np.argmax(final_probs, axis=-1)[0])
        best_text = None
        if candidate_strings is not None and len(candidate_strings) == K:
            best_text = candidate_strings[best_idx]

        ranked_actions: List[ScoredActionDTO] = []
        if candidate_strings is not None and len(candidate_strings) == K:
            probs_flat = final_probs[0]
            logits_flat = logits[0]
            sorted_indices = np.argsort(-probs_flat)

            for rank, idx in enumerate(sorted_indices):
                ranked_actions.append(
                    ScoredActionDTO(
                        rank=rank + 1,
                        action_index=int(idx),
                        action_text=candidate_strings[idx],
                        probability=float(probs_flat[idx]),
                        logit=float(logits_flat[idx])
                    )
                )

        return DecisionOutputDTO(
            probabilities=final_probs,
            logits=logits,
            best_action_idx=best_idx,
            best_action_text=best_text,
            ranked_actions=ranked_actions
        )