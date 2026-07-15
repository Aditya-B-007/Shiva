from __future__ import annotations

import json
from typing import Any, Dict, List

from src.user.constants import BlockCategory
from src.user.graph.workflow_graph import WorkflowGraph


class GraphSerializer:
    def to_dict(self, graph: WorkflowGraph, validate: bool = True) -> Dict[str, Any]:
        if validate:
            graph.validate()

        blocks = [block.to_json() for block in graph.blocks()]
        edges = [edge.to_json() for edge in graph.edges()]

        return {
            "workflow": graph.metadata_dto().to_json(),
            "blocks": blocks,
            "edges": edges,
            "raw_prompt": self._build_raw_prompt(blocks, edges),
        }

    def to_json(self, graph: WorkflowGraph, validate: bool = True, indent: int = 2) -> str:
        return json.dumps(self.to_dict(graph, validate=validate), indent=indent)

    def _build_raw_prompt(
        self,
        blocks: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "inputs": self._blocks_by_category(blocks, BlockCategory.INPUT),
            "decisions": self._blocks_by_category(blocks, BlockCategory.DECISION),
            "outputs": self._blocks_by_category(blocks, BlockCategory.OUTPUT),
            "connections": edges,
        }

    def _blocks_by_category(
        self,
        blocks: List[Dict[str, Any]],
        category: BlockCategory,
    ) -> List[Dict[str, Any]]:
        return [block for block in blocks if block["category"] == category.value]
