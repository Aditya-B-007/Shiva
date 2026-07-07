from __future__ import annotations

from ..graph.MemoryGraph import MemoryGraph
from ..graph.MemoryNode import MemoryStatus, clamp_unit


class ForgettingModel:
    def apply(self, graph: MemoryGraph) -> tuple[str, ...]:
        changed: list[str] = []
        for node in graph.nodes:
            if node.status == MemoryStatus.PRUNED:
                continue
            node.recency = clamp_unit(node.recency - 0.04)
            weakness = (1.0 - node.strength + 1.0 - node.recency + 1.0 - node.activation) / 3.0
            if weakness >= 0.80:
                node.status = MemoryStatus.PRUNED
                node.activation = 0.0
                node.strength = min(node.strength, 0.05)
                changed.append(node.id)
            elif weakness >= 0.55:
                node.status = MemoryStatus.DORMANT
                node.activation = clamp_unit(node.activation - 0.10)
                changed.append(node.id)
        return tuple(changed)
