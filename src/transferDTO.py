from __future__ import annotations
import torch
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any

@dataclass(slots=True)
class Tokens:
    """
    Encapsulates a group of tokens and their associated feature names.
    """
    values: torch.Tensor  # shape: (batch_size, num_features, vector_size)
    names: List[str]      # names of the features in this group


@dataclass(slots=True)
class TokenBundle:
    """
    Encapsulates multiple Token groups (e.g. numerical, categorical, external)
    and provides utility properties to easily merge them.
    """
    groups: Dict[str, Tokens]
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def tensor(self) -> torch.Tensor:
        """
        Concatenates and returns all group token tensors along sequence dimension (dim 1).
        """
        tensors = [g.values for g in self.groups.values() if g.values.size(1) > 0]
        if not tensors:
            raise ValueError("No active tokens found in TokenBundle")
        # Ensure they all share the same device and batch dimension
        return torch.cat(tensors, dim=1)

    @property
    def names(self) -> List[str]:
        """
        Returns a flat list of all feature names across all groups in order.
        """
        feature_names = []
        for g in self.groups.values():
            feature_names.extend(g.names)
        return feature_names


@dataclass(slots=True)
class Latent:
    """
    Carries the pooled latent state vector and contextualized representations for each feature.
    """
    vector: torch.Tensor            # Pooled representation (e.g., shape: (batch_size, vector_size))
    features: Dict[str, torch.Tensor]  # Maps each feature name to its post-attention encoded tensor
    timestamp: datetime = field(default_factory=datetime.utcnow)
