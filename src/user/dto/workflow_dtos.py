from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict


@dataclass(frozen=True, slots=True)
class WorkflowMetadataDTO:
    workflow_id: str
    name: str = "User Workflow"
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.workflow_id,
            "name": self.name,
            "version": self.version,
            "metadata": dict(self.metadata),
            "created_at": self.created_at.isoformat(),
        }
