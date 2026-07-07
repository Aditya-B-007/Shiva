from __future__ import annotations

from dataclasses import dataclass, field

from .MemoryNodeDTO import MemoryEdgeDTO, MemoryNodeDTO


@dataclass(frozen=True, slots=True)
class MemoryGraphDTO:
    nodes: tuple[MemoryNodeDTO, ...] = field(default_factory=tuple)
    edges: tuple[MemoryEdgeDTO, ...] = field(default_factory=tuple)
