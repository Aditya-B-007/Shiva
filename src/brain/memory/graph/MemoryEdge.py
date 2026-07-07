from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .MemoryNode import clamp_unit


class AssociationType(str, Enum):
    SIMILARITY = "SIMILARITY"
    TEMPORAL = "TEMPORAL"
    SPATIAL = "SPATIAL"
    CAUSAL = "CAUSAL"
    SEQUENCE = "SEQUENCE"
    GOAL = "GOAL"
    EMOTIONAL = "EMOTIONAL"
    CONTRADICTION = "CONTRADICTION"


@dataclass(slots=True)
class MemoryEdge:
    source: str
    destination: str
    association_strength: float = 0.5
    association_type: AssociationType | str = AssociationType.SIMILARITY
    creation_time: datetime = field(default_factory=datetime.utcnow)
    last_traversed: datetime | None = None
    activation_probability: float = 0.5
    traversal_count: int = 0

    def __post_init__(self) -> None:
        if not self.source or not self.destination:
            raise ValueError("MemoryEdge source and destination cannot be empty.")
        self.association_type = (
            self.association_type.value
            if isinstance(self.association_type, AssociationType)
            else str(self.association_type)
        )
        self.association_strength = clamp_unit(self.association_strength)
        self.activation_probability = clamp_unit(self.activation_probability)
        if self.traversal_count < 0:
            raise ValueError("traversal_count cannot be negative.")

    def mark_traversed(self) -> None:
        self.traversal_count += 1
        self.last_traversed = datetime.utcnow()
