from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..graph.MemoryEdge import AssociationType, MemoryEdge
from ..graph.MemoryGraph import MemoryGraph
from ..graph.MemoryNode import MemoryNode
from .MemoryRepository import MemoryRepository
from .SQLiteMemoryRepository import SQLiteMemoryRepository
from .ChromaMemoryRepository import ChromaMemoryRepository


class HybridMemoryRepository(MemoryRepository):
    """
    Hybrid Memory Repository combining SQLite and ChromaDB.
    - SQLite: Stores graph edges, state transactions, and credit assignment trajectories.
    - ChromaDB: Manages HNSW dense vector indices for semantic vector search.
    """

    def __init__(
        self,
        database_path: str | Path = "./memory.db",
        chroma_directory: str | Path = "./chroma_db",
    ) -> None:
        self.sqlite_repo = SQLiteMemoryRepository(database_path)
        try:
            self.chroma_repo: Optional[ChromaMemoryRepository] = ChromaMemoryRepository(chroma_directory)
        except Exception:
            self.chroma_repo = None

    def save_node(self, node: MemoryNode) -> None:
        # Dual-write node to both SQLite and ChromaDB
        self.sqlite_repo.save_node(node)
        if self.chroma_repo is not None:
            try:
                self.chroma_repo.save_node(node)
            except Exception:
                pass

    def save_edge(self, edge: MemoryEdge) -> None:
        # Save topological graph edge to SQLite
        self.sqlite_repo.save_edge(edge)
        if self.chroma_repo is not None:
            try:
                self.chroma_repo.save_edge(edge)
            except Exception:
                pass

    def load_node(self, node_id: str) -> MemoryNode | None:
        return self.sqlite_repo.load_node(node_id)

    def load_edge(
        self,
        source: str,
        destination: str,
        association_type: AssociationType | str,
    ) -> MemoryEdge | None:
        return self.sqlite_repo.load_edge(source, destination, association_type)

    def load_graph(self) -> MemoryGraph:
        # Load complete graph topology from SQLite
        return self.sqlite_repo.load_graph()

    def save_graph(self, graph: MemoryGraph) -> None:
        self.sqlite_repo.save_graph(graph)
        if self.chroma_repo is not None:
            try:
                self.chroma_repo.save_graph(graph)
            except Exception:
