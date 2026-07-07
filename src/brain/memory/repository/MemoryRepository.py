from __future__ import annotations

from abc import ABC, abstractmethod

from ..graph.MemoryEdge import AssociationType, MemoryEdge
from ..graph.MemoryGraph import MemoryGraph
from ..graph.MemoryNode import MemoryNode


class MemoryRepository(ABC):
    @abstractmethod
    def save_node(self, node: MemoryNode) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_edge(self, edge: MemoryEdge) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_node(self, node_id: str) -> MemoryNode | None:
        raise NotImplementedError

    @abstractmethod
    def load_edge(
        self,
        source: str,
        destination: str,
        association_type: AssociationType | str,
    ) -> MemoryEdge | None:
        raise NotImplementedError

    @abstractmethod
    def load_graph(self) -> MemoryGraph:
        raise NotImplementedError

    @abstractmethod
    def save_graph(self, graph: MemoryGraph) -> None:
        raise NotImplementedError
