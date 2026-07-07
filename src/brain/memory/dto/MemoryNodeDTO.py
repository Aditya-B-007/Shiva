from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class MemoryNodeDTO:
    id: str
    raw_content: Any
    summary: str
    modality: str
    semantic_type: str
    activation: float
    strength: float
    emotional_salience: float
    identity_relevance: float
    recency: float
    status: str
    access_count: int
    created_at: datetime
    last_accessed: datetime | None = None
    context_signature: str = ""


@dataclass(frozen=True, slots=True)
class MemoryEdgeDTO:
    source: str
    destination: str
    association_strength: float
    association_type: str
    activation_probability: float
    traversal_count: int
    creation_time: datetime
    last_traversed: datetime | None = None
    properties: dict[str, Any] = field(default_factory=dict)
