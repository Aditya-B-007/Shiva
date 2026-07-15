from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class EdgeDTO:
    edge_id: str
    source_block_id: str
    target_block_id: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None
    condition_label: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "id": self.edge_id,
            "source_block_id": self.source_block_id,
            "target_block_id": self.target_block_id,
        }
        optional_fields = {
            "source_handle": self.source_handle,
            "target_handle": self.target_handle,
            "condition_label": self.condition_label,
        }
        payload.update(
            {
                field_name: value
                for field_name, value in optional_fields.items()
                if value is not None
            }
        )
        return payload
