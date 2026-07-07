from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4


class MemoryModality(str, Enum):
    TEXT = "TEXT"
    IMAGE = "IMAGE"
    AUDIO = "AUDIO"
    ACTION = "ACTION"
    SENSOR = "SENSOR"


class MemoryType(str, Enum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"
    GOAL = "goal"
    BELIEF = "belief"


class MemoryStatus(str, Enum):
    NEW = "NEW"
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    PRUNED = "PRUNED"


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(slots=True)
class MemoryNode:
    raw_content: Any
    summary: str
    modality: MemoryModality | str = MemoryModality.TEXT
    semantic_type: MemoryType | str = MemoryType.EPISODIC
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    activation: float = 0.0
    strength: float = 0.5
    emotional_salience: float = 0.0
    identity_relevance: float = 0.0
    recency: float = 1.0
    status: MemoryStatus | str = MemoryStatus.NEW
    access_count: int = 0
    last_accessed: datetime | None = None
    context_signature: str = ""

    def __post_init__(self) -> None:
        self.modality = MemoryModality(self.modality)
        self.semantic_type = MemoryType(self.semantic_type)
        self.status = MemoryStatus(self.status)
        self.activation = clamp_unit(self.activation)
        self.strength = clamp_unit(self.strength)
        self.emotional_salience = clamp_unit(self.emotional_salience)
        self.identity_relevance = clamp_unit(self.identity_relevance)
        self.recency = clamp_unit(self.recency)
        if self.access_count < 0:
            raise ValueError("access_count cannot be negative.")
        if not self.id:
            raise ValueError("MemoryNode.id cannot be empty.")

    def mark_accessed(self) -> None:
        self.access_count += 1
        self.last_accessed = datetime.utcnow()
        if self.status == MemoryStatus.NEW:
            self.status = MemoryStatus.ACTIVE
