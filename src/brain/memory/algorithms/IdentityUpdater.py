from __future__ import annotations

from ..graph.MemoryGraph import MemoryGraph
from ..graph.MemoryNode import MemoryStatus, clamp_unit


class IdentityUpdater:
    def update(self, graph: MemoryGraph) -> tuple[str, ...]:
        updated: list[str] = []
        for node in graph.nodes:
            if node.status == MemoryStatus.PRUNED:
                continue
            text = f"{node.summary} {node.raw_content}".lower()
            self_signal = any(token in text for token in ("i ", "me ", "my ", "self", "shiva"))
            if self_signal:
                node.identity_relevance = clamp_unit(node.identity_relevance + 0.10)
                node.strength = clamp_unit(node.strength + 0.04)
                updated.append(node.id)
            elif node.identity_relevance > 0.0:
                node.identity_relevance = clamp_unit(node.identity_relevance - 0.02)
        return tuple(updated)
