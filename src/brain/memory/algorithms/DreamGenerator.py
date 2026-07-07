from __future__ import annotations
from ..graph.MemoryGraph import MemoryGraph
from ..graph.MemoryNode import MemoryNode, MemoryStatus


class DreamGenerator:
    def generate(self, graph: MemoryGraph, max_length: int = 8) -> tuple[MemoryNode, ...]:
        candidates = [
            node for node in graph.nodes
            if node.status != MemoryStatus.PRUNED
        ]
        if not candidates or max_length <= 0:
            return ()
        start = max(candidates, key=self._node_weight)
        sequence = [start]
        seen = {start.id}
        current = start
        while len(sequence) < max_length:
            next_node = self._best_neighbor(graph, current.id, seen)
            if next_node is None:
                break
            sequence.append(next_node)
            seen.add(next_node.id)
            current = next_node
        return tuple(sequence)

    def _best_neighbor(
        self,
        graph: MemoryGraph,
        node_id: str,
        seen: set[str],
    ) -> MemoryNode | None:
        neighbors = [
            node for node in graph.neighbors(node_id)
            if node.id not in seen and node.status != MemoryStatus.PRUNED
        ]
        if not neighbors:
            return None
        return max(neighbors, key=self._node_weight)

    def _node_weight(self, node: MemoryNode) -> float:
        return (
            node.activation * 0.30
            + node.emotional_salience * 0.25
            + node.identity_relevance * 0.20
            + node.recency * 0.15
            + node.strength * 0.10
        )
