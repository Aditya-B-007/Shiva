from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from src.user.constants import OutputFormat

@dataclass(slots=True)
class WorkflowBlock:
    """
    Represents a single drag-and-drop block connected in the user's workflow.
    """
    device: str
    arguments: Dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class Workflow:
    """
    Represents the full serialized workflow sent from the frontend.
    """
    query: str
    blocks: List[WorkflowBlock] = field(default_factory=list)
    output_format: OutputFormat = OutputFormat.TEXT
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> Workflow:
        """
        Parses a workflow dictionary safely.
        """
        raw_blocks = data.get("blocks", [])
        blocks = [
            WorkflowBlock(
                device=b.get("device", ""),
                arguments=dict(b.get("arguments", {}))
            )
            for b in raw_blocks
        ]
        
        raw_format = data.get("output_format", "text")
        try:
            output_format = OutputFormat(raw_format.lower())
        except ValueError:
            output_format = OutputFormat.TEXT
            
        return cls(
            query=data.get("query", ""),
            blocks=blocks,
            output_format=output_format,
            metadata=dict(data.get("metadata", {}))
        )
