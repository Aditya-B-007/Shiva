from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Dict, Tuple

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore

from ..graph.MemoryEdge import AssociationType, MemoryEdge
from ..graph.MemoryGraph import MemoryGraph
from ..graph.MemoryNode import MemoryModality, MemoryNode, MemoryStatus, MemoryType
from .MemoryRepository import MemoryRepository


class ChromaMemoryRepository(MemoryRepository):
    """
    ChromaDB-backed vector memory repository for Shiva.ai.
    Indexes memory node text embeddings for dense vector similarity search.
    """

    def __init__(self, persist_directory: str | Path = "./chroma_db") -> None:
        if chromadb is None:
            raise ImportError(
                "ChromaDB is required for ChromaMemoryRepository. Install it via 'pip install chromadb'."
            )
        self._persist_dir = str(persist_directory)
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self._persist_dir)

        # Vector collection for semantic similarity queries on memory nodes
        self._nodes_col = self._client.get_or_create_collection(
            name="shiva_nodes",
            metadata={"hnsw:space": "cosine"}
        )

        # Graph edges collection
        self._edges_col = self._client.get_or_create_collection(
            name="shiva_edges"
        )

    def _build_document_text(self, node: MemoryNode) -> str:
        """Serializes memory node attributes into a composite document for dense vector embedding."""
        raw_str = json.dumps(node.raw_content, default=str) if isinstance(node.raw_content, (dict, list)) else str(node.raw_content)
        return f"{node.summary} | Context: {node.context_signature} | Raw: {raw_str}"

    def save_node(self, node: MemoryNode) -> None:
        document_text = self._build_document_text(node)
        metadata = {
            "created_at": self._datetime_to_text(node.created_at) or "",
            "modality": node.modality.value if hasattr(node.modality, "value") else str(node.modality),
            "semantic_type": node.semantic_type.value if hasattr(node.semantic_type, "value") else str(node.semantic_type),
            "activation": float(node.activation),
            "strength": float(node.strength),
            "emotional_salience": float(node.emotional_salience),
            "identity_relevance": float(node.identity_relevance),
            "recency": float(node.recency),
            "status": node.status.value if hasattr(node.status, "value") else str(node.status),
            "access_count": int(node.access_count),
            "last_accessed": self._datetime_to_text(node.last_accessed) or "",
            "context_signature": str(node.context_signature or ""),
            "summary": str(node.summary or ""),
            "raw_content_json": json.dumps(node.raw_content, default=str),
        }

        self._nodes_col.upsert(
            ids=[node.id],
            documents=[document_text],
            metadatas=[metadata]
        )

    def save_edge(self, edge: MemoryEdge) -> None:
        edge_id = f"{edge.source}___{edge.destination}___{self._association_value(edge.association_type)}"
        metadata = {
            "source": str(edge.source),
            "destination": str(edge.destination),
            "association_type": self._association_value(edge.association_type),
            "association_strength": float(edge.association_strength),
            "creation_time": self._datetime_to_text(edge.creation_time) or "",
            "last_traversed": self._datetime_to_text(edge.last_traversed) or "",
            "activation_probability": float(edge.activation_probability),
            "traversal_count": int(edge.traversal_count),
        }

        self._edges_col.upsert(
            ids=[edge_id],
            documents=[edge_id],
            metadatas=[metadata]
        )

    def load_node(self, node_id: str) -> MemoryNode | None:
        result = self._nodes_col.get(ids=[node_id])
        if not result["ids"]:
            return None
        return self._node_from_chroma(result["ids"][0], result["metadatas"][0])

    def load_edge(
        self,
        source: str,
        destination: str,
        association_type: AssociationType | str,
    ) -> MemoryEdge | None:
        edge_id = f"{source}___{destination}___{self._association_value(association_type)}"
        result = self._edges_col.get(ids=[edge_id])
        if not result["ids"]:
            return None
        return self._edge_from_chroma(result["metadatas"][0])

    def load_graph(self) -> MemoryGraph:
        nodes_data = self._nodes_col.get()
        nodes = []
        if nodes_data and nodes_data["ids"]:
            for node_id, meta in zip(nodes_data["ids"], nodes_data["metadatas"]):
                nodes.append(self._node_from_chroma(node_id, meta))

        graph = MemoryGraph(nodes=nodes)

        edges_data = self._edges_col.get()
        if edges_data and edges_data["ids"]:
            for meta in edges_data["metadatas"]:
                graph.connect_edge(self._edge_from_chroma(meta))

        return graph

    def save_graph(self, graph: MemoryGraph) -> None:
        for node in graph.nodes:
            self.save_node(node)
        for edge in graph.edges:
            self.save_edge(edge)

    def vector_search(self, query_text: str, limit: int = 5) -> Dict[str, float]:
        """
        Executes dense vector search in ChromaDB.
        Returns a mapping of node_id -> normalized cosine similarity score (0.0 to 1.0).
        """
        count = self._nodes_col.count()
        if count == 0:
            return {}
        n_results = min(limit, count)
        results = self._nodes_col.query(
            query_texts=[query_text],
            n_results=n_results,
            include=["distances", "metadatas"]
        )
        scores: Dict[str, float] = {}
        if results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)
            for node_id, dist in zip(ids, distances):
                # Convert cosine distance to similarity score in range [0, 1]
                sim_score = max(0.0, min(1.0, 1.0 - dist))
                scores[node_id] = sim_score
        return scores

    def _node_from_chroma(self, node_id: str, meta: dict[str, Any]) -> MemoryNode:
        return MemoryNode(
            id=node_id,
            created_at=self._datetime_from_text(meta.get("created_at")),  # type: ignore
            raw_content=json.loads(meta.get("raw_content_json", "{}")),
            summary=meta.get("summary", ""),
            modality=MemoryModality(meta["modality"]) if "modality" in meta else MemoryModality.TEXT,
            semantic_type=MemoryType(meta["semantic_type"]) if "semantic_type" in meta else MemoryType.EPISODIC,
            activation=float(meta.get("activation", 0.0)),
            strength=float(meta.get("strength", 1.0)),
            emotional_salience=float(meta.get("emotional_salience", 0.0)),
            identity_relevance=float(meta.get("identity_relevance", 0.0)),
            recency=float(meta.get("recency", 1.0)),
            status=MemoryStatus(meta["status"]) if "status" in meta else MemoryStatus.ACTIVE,
            access_count=int(meta.get("access_count", 0)),
            last_accessed=self._datetime_from_text(meta.get("last_accessed")),
            context_signature=meta.get("context_signature", ""),
        )

    def _edge_from_chroma(self, meta: dict[str, Any]) -> MemoryEdge:
        return MemoryEdge(
            source=meta["source"],
            destination=meta["destination"],
            association_type=AssociationType(meta["association_type"]) if "association_type" in meta else AssociationType.SIMILARITY,
            association_strength=float(meta.get("association_strength", 0.5)),
            creation_time=self._datetime_from_text(meta.get("creation_time")),  # type: ignore
            last_traversed=self._datetime_from_text(meta.get("last_traversed")),
            activation_probability=float(meta.get("activation_probability", 0.5)),
            traversal_count=int(meta.get("traversal_count", 0)),
        )

    def _datetime_to_text(self, value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    def _datetime_from_text(self, value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    def _association_value(self, association_type: AssociationType | str) -> str:
        return association_type.value if isinstance(association_type, AssociationType) else str(association_type)
