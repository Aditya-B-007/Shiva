from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True, slots=True)
class ConditionDTO:
    left_operand: str
    operator: str
    right_operand: str

    def to_json(self) -> Dict[str, str]:
        return {
            "left_operand": self.left_operand,
            "operator": self.operator,
            "right_operand": self.right_operand,
        }


@dataclass(frozen=True, slots=True)
class BlockSchemaDTO:
    category: str
    block_type: str
    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    condition: Optional[ConditionDTO] = None

    def to_json(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "category": self.category,
            "type": self.block_type,
            "name": self.name,
            "config": dict(self.config),
        }
        if self.condition is not None:
            payload["condition"] = self.condition.to_json()
        return payload
