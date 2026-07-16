from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


class OutputFormat(str, Enum):
    TEXT = "text"
    VOICE = "voice"


@dataclass(slots=True)
class WorkflowBlock:
    device: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowBlock":
        return cls(
            device=str(data.get("device", "")),
            arguments=dict(data.get("arguments", {})),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "arguments": dict(self.arguments),
        }


@dataclass(slots=True)
class GraphBlock:
    id: str
    type: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    condition: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "arguments": dict(self.arguments),
            "condition": dict(self.condition),
        }


@dataclass(slots=True)
class GraphEdge:
    id: str
    source_block_id: str
    target_block_id: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_block_id": self.source_block_id,
            "target_block_id": self.target_block_id,
        }


@dataclass(slots=True)
class WorkflowGraph:
    blocks: List[GraphBlock] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "blocks": [block.to_json() for block in self.blocks],
            "edges": [edge.to_json() for edge in self.edges],
        }


@dataclass(slots=True)
class WorkflowRequest:
    query: str
    blocks: List[WorkflowBlock] = field(default_factory=list)
    output_format: OutputFormat = OutputFormat.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowRequest":
        raw_format = str(data.get("output_format", OutputFormat.TEXT.value)).lower()
        try:
            output_format = OutputFormat(raw_format)
        except ValueError:
            output_format = OutputFormat.TEXT

        return cls(
            query=str(data.get("query", "")),
            blocks=[
                WorkflowBlock.from_dict(block)
                for block in data.get("blocks", [])
                if isinstance(block, dict)
            ],
            output_format=output_format,
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "blocks": [block.to_json() for block in self.blocks],
            "output_format": self.output_format.value,
            "metadata": dict(self.metadata),
        }
