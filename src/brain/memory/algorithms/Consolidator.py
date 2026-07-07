from __future__ import annotations

from ..graph.MemoryGraph import MemoryGraph
from ..graph.MemoryNode import MemoryStatus, clamp_unit


class Consolidator:
    def consolidate(self, graph: MemoryGraph) -> tuple[str, ...]:
        consolidated: list[str] = []
        for node in graph.nodes:
            importance = (node.activation + node.emotional_salience + node.identity_relevance) / 3.0
            if importance >= 0.55 and node.status != MemoryStatus.PRUNED:
                node.strength = clamp_unit(node.strength + 0.08 * importance)
                node.status = MemoryStatus.ACTIVE
                consolidated.append(node.id)
            elif node.status != MemoryStatus.PRUNED:
                node.strength = clamp_unit(node.strength - 0.03)
        for edge in graph.edges:
            source = graph.get_node(edge.source)
            destination = graph.get_node(edge.destination)
            if source is None or destination is None:
                continue
            endpoint_importance = (source.strength + destination.strength) / 2.0
            delta = 0.04 if endpoint_importance >= 0.55 else -0.02
            edge.association_strength = clamp_unit(edge.association_strength + delta)
            edge.activation_probability = clamp_unit(edge.activation_probability + delta / 2.0)
        return tuple(consolidated)
