from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.user.dto import EdgeDTO


@dataclass(frozen=True, slots=True)
class EdgeConnection:
    edge_id: str
    source_block_id: str
    target_block_id: str
    source_handle: Optional[str] = None
    target_handle: Optional[str] = None
    condition_label: Optional[str] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        required_values = {
            "edge_id": self.edge_id,
            "source_block_id": self.source_block_id,
            "target_block_id": self.target_block_id,
        }
        for field_name, value in required_values.items():
            if not value or not value.strip():
                raise ValueError(f"{field_name} is required.")
        if self.source_block_id == self.target_block_id:
            raise ValueError("source_block_id and target_block_id must be different.")

    def to_dto(self) -> EdgeDTO:
        return EdgeDTO(
            edge_id=self.edge_id,
            source_block_id=self.source_block_id,
            target_block_id=self.target_block_id,
            source_handle=self.source_handle,
            target_handle=self.target_handle,
            condition_label=self.condition_label,
        )

    def to_json(self) -> Dict[str, Any]:
        return self.to_dto().to_json()
