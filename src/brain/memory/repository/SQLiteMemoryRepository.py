from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import json
import sqlite3

from ..graph.MemoryEdge import AssociationType, MemoryEdge
from ..graph.MemoryGraph import MemoryGraph
from ..graph.MemoryNode import MemoryModality, MemoryNode, MemoryStatus, MemoryType
from .MemoryRepository import MemoryRepository


class SQLiteMemoryRepository(MemoryRepository):
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self._database_path))
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def save_node(self, node: MemoryNode) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO memory_nodes (
                    id, created_at, raw_content_json, summary, modality, semantic_type,
                    activation, strength, emotional_salience, identity_relevance,
                    recency, status, access_count, last_accessed, context_signature,
                    properties_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    raw_content_json = excluded.raw_content_json,
                    summary = excluded.summary,
                    modality = excluded.modality,
                    semantic_type = excluded.semantic_type,
                    activation = excluded.activation,
                    strength = excluded.strength,
                    emotional_salience = excluded.emotional_salience,
                    identity_relevance = excluded.identity_relevance,
                    recency = excluded.recency,
                    status = excluded.status,
                    access_count = excluded.access_count,
                    last_accessed = excluded.last_accessed,
                    context_signature = excluded.context_signature,
                    properties_json = excluded.properties_json
                """,
                self._node_row(node),
            )

    def save_edge(self, edge: MemoryEdge) -> None:
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO memory_edges (
                    source, destination, association_type, association_strength,
                    creation_time, last_traversed, activation_probability,
                    traversal_count, properties_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, destination, association_type) DO UPDATE SET
                    association_strength = excluded.association_strength,
                    last_traversed = excluded.last_traversed,
                    activation_probability = excluded.activation_probability,
                    traversal_count = excluded.traversal_count,
                    properties_json = excluded.properties_json
                """,
                self._edge_row(edge),
            )

    def load_node(self, node_id: str) -> MemoryNode | None:
        row = self._connection.execute(
            "SELECT * FROM memory_nodes WHERE id = ?",
            (node_id,),
        ).fetchone()
        return self._node_from_row(row) if row is not None else None

    def load_edge(
        self,
        source: str,
        destination: str,
        association_type: AssociationType | str,
    ) -> MemoryEdge | None:
        row = self._connection.execute(
            """
            SELECT * FROM memory_edges
            WHERE source = ? AND destination = ? AND association_type = ?
            """,
            (source, destination, self._association_value(association_type)),
        ).fetchone()
        return self._edge_from_row(row) if row is not None else None

    def load_graph(self) -> MemoryGraph:
        nodes = [
            self._node_from_row(row)
            for row in self._connection.execute("SELECT * FROM memory_nodes")
        ]
        graph = MemoryGraph(nodes=nodes)
        for row in self._connection.execute("SELECT * FROM memory_edges"):
            edge = self._edge_from_row(row)
            if graph.get_node(edge.source) is not None and graph.get_node(edge.destination) is not None:
                graph.connect_edge(edge)
        return graph

    def save_graph(self, graph: MemoryGraph) -> None:
        for node in graph.nodes:
            self.save_node(node)
        for edge in graph.edges:
            self.save_edge(edge)

    def close(self) -> None:
        self._connection.close()

    def _initialize_schema(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    raw_content_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    semantic_type TEXT NOT NULL,
                    activation REAL NOT NULL,
                    strength REAL NOT NULL,
                    emotional_salience REAL NOT NULL,
                    identity_relevance REAL NOT NULL,
                    recency REAL NOT NULL,
                    status TEXT NOT NULL,
                    access_count INTEGER NOT NULL,
                    last_accessed TEXT,
                    context_signature TEXT NOT NULL,
                    properties_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_edges (
                    source TEXT NOT NULL,
                    destination TEXT NOT NULL,
                    association_type TEXT NOT NULL,
                    association_strength REAL NOT NULL,
                    creation_time TEXT NOT NULL,
                    last_traversed TEXT,
                    activation_probability REAL NOT NULL,
                    traversal_count INTEGER NOT NULL,
                    properties_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (source, destination, association_type),
                    FOREIGN KEY (source) REFERENCES memory_nodes(id),
                    FOREIGN KEY (destination) REFERENCES memory_nodes(id)
                )
                """
            )

    def _node_row(self, node: MemoryNode) -> tuple[Any, ...]:
        return (
            node.id,
            self._datetime_to_text(node.created_at),
            self._json_dump(node.raw_content),
            node.summary,
            MemoryModality(node.modality).value,
            MemoryType(node.semantic_type).value,
            node.activation,
            node.strength,
            node.emotional_salience,
            node.identity_relevance,
            node.recency,
            MemoryStatus(node.status).value,
            node.access_count,
            self._datetime_to_text(node.last_accessed),
            node.context_signature,
            "{}",
        )

    def _edge_row(self, edge: MemoryEdge) -> tuple[Any, ...]:
        return (
            edge.source,
            edge.destination,
            self._association_value(edge.association_type),
            edge.association_strength,
            self._datetime_to_text(edge.creation_time),
            self._datetime_to_text(edge.last_traversed),
            edge.activation_probability,
            edge.traversal_count,
            "{}",
        )

    def _node_from_row(self, row: sqlite3.Row) -> MemoryNode:
        return MemoryNode(
            id=row["id"],
            created_at=self._datetime_from_text(row["created_at"]), # type: ignore
            raw_content=json.loads(row["raw_content_json"]),
            summary=row["summary"],
            modality=row["modality"],
            semantic_type=row["semantic_type"],
            activation=row["activation"],
            strength=row["strength"],
            emotional_salience=row["emotional_salience"],
            identity_relevance=row["identity_relevance"],
            recency=row["recency"],
            status=row["status"],
            access_count=row["access_count"],
            last_accessed=self._datetime_from_text(row["last_accessed"]),
            context_signature=row["context_signature"],
        )

    def _edge_from_row(self, row: sqlite3.Row) -> MemoryEdge:
        return MemoryEdge(
            source=row["source"],
            destination=row["destination"],
            association_type=row["association_type"],
            association_strength=row["association_strength"],
            creation_time=self._datetime_from_text(row["creation_time"]), # type: ignore
            last_traversed=self._datetime_from_text(row["last_traversed"]),
            activation_probability=row["activation_probability"],
            traversal_count=row["traversal_count"],
        )

    def _json_dump(self, value: Any) -> str:
        try:
            return json.dumps(value, default=str)
        except TypeError:
            return json.dumps(str(value))

    def _datetime_to_text(self, value: datetime | None) -> str | None:
        return value.isoformat() if value is not None else None

    def _datetime_from_text(self, value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    def _association_value(self, association_type: AssociationType | str) -> str:
        return association_type.value if isinstance(association_type, AssociationType) else str(association_type)
