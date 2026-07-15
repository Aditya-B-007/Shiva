from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from src.user.blocks import BaseBlock
from src.user.constants import BlockCategory
from src.user.dto import WorkflowMetadataDTO
from src.user.graph.edge_connections import EdgeConnection


@dataclass(slots=True)
class WorkflowGraph:
    workflow_id: str
    name: str = "User Workflow"
    version: str = "1.0"
    metadata: Dict[str, object] = field(default_factory=dict)
    _blocks: Dict[str, BaseBlock] = field(default_factory=dict, init=False)
    _edges: Dict[str, EdgeConnection] = field(default_factory=dict, init=False)

    def add_block(self, block: BaseBlock) -> None:
        if not isinstance(block, BaseBlock):
            raise TypeError("block must inherit from BaseBlock.")
        if block.block_id in self._blocks:
            raise ValueError(f"Block '{block.block_id}' already exists.")
        self._blocks[block.block_id] = block

    def get_block(self, block_id: str) -> Optional[BaseBlock]:
        return self._blocks.get(block_id)

    def remove_block(self, block_id: str) -> None:
        if block_id not in self._blocks:
            raise KeyError(f"Block '{block_id}' does not exist.")
        connected_edges = [
            edge_id
            for edge_id, edge in self._edges.items()
            if edge.source_block_id == block_id or edge.target_block_id == block_id
        ]
        for edge_id in connected_edges:
            del self._edges[edge_id]
        del self._blocks[block_id]

    def blocks(self) -> List[BaseBlock]:
        return list(self._blocks.values())

    def add_edge(self, edge: EdgeConnection) -> None:
        if not isinstance(edge, EdgeConnection):
            raise TypeError("edge must be an EdgeConnection.")
        if edge.edge_id in self._edges:
            raise ValueError(f"Edge '{edge.edge_id}' already exists.")
        if edge.source_block_id not in self._blocks:
            raise ValueError(f"Source block '{edge.source_block_id}' does not exist.")
        if edge.target_block_id not in self._blocks:
            raise ValueError(f"Target block '{edge.target_block_id}' does not exist.")
        self._edges[edge.edge_id] = edge

    def get_edge(self, edge_id: str) -> Optional[EdgeConnection]:
        return self._edges.get(edge_id)

    def remove_edge(self, edge_id: str) -> None:
        if edge_id not in self._edges:
            raise KeyError(f"Edge '{edge_id}' does not exist.")
        del self._edges[edge_id]

    def edges(self) -> List[EdgeConnection]:
        return list(self._edges.values())

    def incoming_edges(self, block_id: str) -> List[EdgeConnection]:
        return [edge for edge in self._edges.values() if edge.target_block_id == block_id]

    def outgoing_edges(self, block_id: str) -> List[EdgeConnection]:
        return [edge for edge in self._edges.values() if edge.source_block_id == block_id]

    def validate(self) -> None:
        if not self.workflow_id or not self.workflow_id.strip():
            raise ValueError("workflow_id is required.")
        if not self._blocks:
            raise ValueError("WorkflowGraph must contain at least one block.")

        categories = [block.category for block in self._blocks.values()]
        if BlockCategory.INPUT not in categories:
            raise ValueError("WorkflowGraph must contain at least one input block.")
        if BlockCategory.OUTPUT not in categories:
            raise ValueError("WorkflowGraph must contain at least one output block.")

        for edge in self._edges.values():
            if edge.source_block_id not in self._blocks:
                raise ValueError(f"Edge '{edge.edge_id}' has an unknown source block.")
            if edge.target_block_id not in self._blocks:
                raise ValueError(f"Edge '{edge.edge_id}' has an unknown target block.")

    def metadata_dto(self) -> WorkflowMetadataDTO:
        return WorkflowMetadataDTO(
            workflow_id=self.workflow_id,
            name=self.name,
            version=self.version,
            metadata=self.metadata,
        )

    def add_blocks(self, blocks: Iterable[BaseBlock]) -> None:
        for block in blocks:
            self.add_block(block)

    def add_edges(self, edges: Iterable[EdgeConnection]) -> None:
        for edge in edges:
            self.add_edge(edge)
