from __future__ import annotations

from dataclasses import dataclass, field

from .MemoryNodeDTO import MemoryNodeDTO


@dataclass(frozen=True, slots=True)
class RetrievalDTO:
    query: object
    memories: tuple[MemoryNodeDTO, ...] = field(default_factory=tuple)
    confidence: float = 0.0
