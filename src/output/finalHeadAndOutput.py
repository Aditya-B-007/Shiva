import numpy as np
from typing import List, Optional, Union
try:
    from src.config.dtos import (
        CandidateOptionDTO,
        OptionMatrixRequestDTO,
        OptionScoreDTO,
        SelectedActionDTO,
        DecisionMetricsDTO,
        DecisionDistributionDTO,
        FinalDecisionOutputDTO,
        DecisionResultDTO
    )
except ImportError:
    from Shiva.src.config.dtos import (
        CandidateOptionDTO,
        OptionMatrixRequestDTO,
        OptionScoreDTO,
        SelectedActionDTO,
        DecisionMetricsDTO,
        DecisionDistributionDTO,
        FinalDecisionOutputDTO,
        DecisionResultDTO
    )

__all__ = [
    "FinalHeadAndOutput",
    "OptionScoreDTO",
    "SelectedActionDTO",
    "DecisionMetricsDTO",
    "DecisionDistributionDTO",
    "FinalDecisionOutputDTO",
    "DecisionResultDTO"
]


class FinalHeadAndOutput:
    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature

    def forward(
        self,
        situation_vector: np.ndarray,
        options_matrix: np.ndarray,
        candidate_strings: Optional[Union[List[str], List[CandidateOptionDTO], OptionMatrixRequestDTO]] = None,
        query_id: Optional[str] = None
    ) -> FinalDecisionOutputDTO:
        assert situation_vector.ndim == 2 and situation_vector.shape[0] == 1, (
            f"Expected situation_vector shape [1, 512], got {situation_vector.shape}"
        )
        assert options_matrix.ndim == 2, (
            f"Expected options_matrix shape [K, 512], got {options_matrix.shape}"
        )
        assert situation_vector.shape[1] == options_matrix.shape[1], (
            f"Dimension mismatch: situation_vector is {situation_vector.shape[1]}-d, "
            f"options_matrix is {options_matrix.shape[1]}-d"
        )

        K = options_matrix.shape[0]
        if K == 0:
            raise ValueError("Candidate options matrix contains zero options (K=0).")

        # 1. Cosine Similarity & Logits: [1, K]
        eps = 1e-12
        z_norm = situation_vector / (np.linalg.norm(situation_vector, ord=2, axis=-1, keepdims=True) + eps)
        cos_similarity = np.matmul(z_norm, options_matrix.T)  # Shape: [1, K]
        logits = cos_similarity / self.temperature            # Shape: [1, K]

        # 2. Numerically stable softmax: [1, K]
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probabilities = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # Shape: [1, K]

        # Flatten 1D views of length K
        probs_flat = probabilities[0]
        logits_flat = logits[0]
        cos_flat = cos_similarity[0]

        # 3. Resolve candidate texts, IDs, and query_id
        resolved_query_id = query_id
        option_texts = [f"Option {i + 1}" for i in range(K)]
        option_ids = [f"opt_{i + 1}" for i in range(K)]

        if isinstance(candidate_strings, OptionMatrixRequestDTO):
            resolved_query_id = resolved_query_id or candidate_strings.query_id
            for i, opt in enumerate(candidate_strings.options[:K]):
                option_texts[i] = opt.text
                option_ids[i] = opt.id
        elif isinstance(candidate_strings, list):
            for i, item in enumerate(candidate_strings[:K]):
                if isinstance(item, CandidateOptionDTO):
                    option_texts[i] = item.text
                    option_ids[i] = item.id
                else:
                    option_texts[i] = str(item)

        # 4. Ranking & Metrics
        sorted_indices = np.argsort(-probs_flat)
        best_idx = int(sorted_indices[0])
        runner_up_idx = int(sorted_indices[1]) if K > 1 else None
        runner_up_prob = float(probs_flat[runner_up_idx]) if runner_up_idx is not None else None
        margin = float(probs_flat[best_idx] - runner_up_prob) if runner_up_prob is not None else 1.0

        # Shannon entropy H = -sum(p * log(p + eps))
        entropy = -float(np.sum(probs_flat * np.log(probs_flat + eps)))
        entropy = max(0.0, entropy)

        # 5. Build Sub-DTOs
        selected_action = SelectedActionDTO(
            rank=1,
            index=best_idx,
            id=option_ids[best_idx],
            text=option_texts[best_idx],
            confidence=float(probs_flat[best_idx]),
            logit=float(logits_flat[best_idx]),
            margin_over_runner_up=margin
        )

        rankings: List[OptionScoreDTO] = [
            OptionScoreDTO(
                rank=rank,
                index=int(idx),
                id=option_ids[idx],
                text=option_texts[idx],
                probability=float(probs_flat[idx]),
                logit=float(logits_flat[idx]),
                cosine_sim=float(cos_flat[idx])
            )
            for rank, idx in enumerate(sorted_indices, start=1)
        ]

        metrics = DecisionMetricsDTO(
            total_options=K,
            temperature=float(self.temperature),
            top_probability=float(probs_flat[best_idx]),
            runner_up_probability=runner_up_prob,
            confidence_margin=margin,
            entropy=entropy
        )

        distribution = DecisionDistributionDTO(
            shape=[1, K],
            probabilities_matrix=probabilities.tolist(),
            logits_matrix=logits.tolist(),
            probabilities=probs_flat.tolist(),
            logits=logits_flat.tolist(),
            cosine_similarities=cos_flat.tolist()
        )

        return FinalDecisionOutputDTO(
            query_id=resolved_query_id,
            selected_action=selected_action,
            rankings=rankings,
            metrics=metrics,
            distribution=distribution
        )