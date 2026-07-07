from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

from .MemoryEdge import AssociationType, MemoryEdge
from .MemoryNode import MemoryNode, MemoryStatus, clamp_unit


class MemoryGraph:
    def __init__(
        self,
        nodes: Iterable[MemoryNode] | None = None,
        edges: Iterable[MemoryEdge] | None = None,
    ) -> None:
        self._nodes: dict[str, MemoryNode] = {}
        self._edges: dict[tuple[str, str, str], MemoryEdge] = {}
        for node in nodes or ():
            self.add_node(node)
        for edge in edges or ():
            self.connect_edge(edge)

    @property
    def nodes(self) -> tuple[MemoryNode, ...]:
        return tuple(self._nodes.values())

    @property
    def edges(self) -> tuple[MemoryEdge, ...]:
        return tuple(self._edges.values())

    def add_node(self, node: MemoryNode) -> MemoryNode:
        if node.id in self._nodes:
            raise ValueError(f"Memory node already exists: {node.id}")
        self._nodes[node.id] = node
        return node

    def connect(
        self,
        source: str,
        destination: str,
        association_strength: float = 0.5,
        association_type: AssociationType | str = AssociationType.SIMILARITY,
        activation_probability: float = 0.5,
    ) -> MemoryEdge:
        edge = MemoryEdge(
            source=source,
            destination=destination,
            association_strength=association_strength,
            association_type=association_type,
            activation_probability=activation_probability,
        )
        return self.connect_edge(edge)

    def connect_edge(self, edge: MemoryEdge) -> MemoryEdge:
        self._require_node(edge.source)
        self._require_node(edge.destination)
        key = self._edge_key(edge.source, edge.destination, edge.association_type)
        self._edges[key] = edge
        return edge

    def remove_edge(
        self,
        source: str,
        destination: str,
        association_type: AssociationType | str | None = None,
    ) -> None:
        if association_type is None:
            keys = [
                key for key in self._edges
                if key[0] == source and key[1] == destination
            ]
            for key in keys:
                del self._edges[key]
            return
        self._edges.pop(self._edge_key(source, destination, association_type), None)

    def activate(self, node_id: str, amount: float = 0.1) -> MemoryNode:
        node = self._require_node(node_id)
        if node.status != MemoryStatus.PRUNED:
            node.activation = clamp_unit(node.activation + amount)
            node.status = MemoryStatus.ACTIVE
        node.mark_accessed()
        return node

    def deactivate(self, node_id: str, amount: float = 0.1) -> MemoryNode:
        node = self._require_node(node_id)
        node.activation = clamp_unit(node.activation - amount)
        if node.activation <= 0.05 and node.status == MemoryStatus.ACTIVE:
            node.status = MemoryStatus.DORMANT
        return node

    def neighbors(
        self,
        node_id: str,
        association_type: AssociationType | str | None = None,
    ) -> tuple[MemoryNode, ...]:
        self._require_node(node_id)
        expected_type = self._association_value(association_type) if association_type else None
        result = []
        for edge in self._edges.values():
            if edge.source != node_id:
                continue
            if expected_type is not None and edge.association_type != expected_type:
                continue
            result.append(self._nodes[edge.destination])
        return tuple(result)

    def traverse(self, start_id: str, depth: int = 2) -> tuple[MemoryNode, ...]:
        self._require_node(start_id)
        if depth < 0:
            raise ValueError("depth cannot be negative.")
        visited: set[str] = set()
        ordered: list[MemoryNode] = []
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        while queue:
            node_id, current_depth = queue.popleft()
            if node_id in visited or current_depth > depth:
                continue
            visited.add(node_id)
            ordered.append(self.activate(node_id, 0.02))
            if current_depth == depth:
                continue
            for edge in self._outgoing_edges(node_id):
                edge.mark_traversed()
                queue.append((edge.destination, current_depth + 1))
        return tuple(ordered)

    def find(self, predicate: Callable[[MemoryNode], bool]) -> tuple[MemoryNode, ...]:
        return tuple(node for node in self._nodes.values() if predicate(node))

    def prune(self, predicate: Callable[[MemoryNode], bool]) -> tuple[MemoryNode, ...]:
        pruned: list[MemoryNode] = []
        for node in self._nodes.values():
            if predicate(node):
                node.status = MemoryStatus.PRUNED
                node.activation = 0.0
                node.strength = min(node.strength, 0.05)
                pruned.append(node)
        return tuple(pruned)

    def get_node(self, node_id: str) -> MemoryNode | None:
        return self._nodes.get(node_id)

    def get_edge(
        self,
        source: str,
        destination: str,
        association_type: AssociationType | str,
    ) -> MemoryEdge | None:
        return self._edges.get(self._edge_key(source, destination, association_type))

    def _outgoing_edges(self, node_id: str) -> list[MemoryEdge]:
        return sorted(
            (edge for edge in self._edges.values() if edge.source == node_id),
            key=lambda edge: edge.association_strength * edge.activation_probability,
            reverse=True,
        )

    def _require_node(self, node_id: str) -> MemoryNode:
        try:
            return self._nodes[node_id]
        except KeyError as exc:
            raise KeyError(f"Unknown memory node: {node_id}") from exc

    def _edge_key(
        self,
        source: str,
        destination: str,
        association_type: AssociationType | str,
    ) -> tuple[str, str, str]:
        return (source, destination, self._association_value(association_type))

    def _association_value(self, association_type: AssociationType | str) -> str:
        return association_type.value if isinstance(association_type, AssociationType) else str(association_type)
