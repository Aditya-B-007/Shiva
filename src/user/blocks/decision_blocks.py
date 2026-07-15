from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

from src.user.blocks.base import BaseBlock
from src.user.constants import BlockCategory, DecisionBlockType
from src.user.dto import ConditionDTO


@dataclass(slots=True)
class BaseDecisionBlock(BaseBlock):
    condition: Optional[ConditionDTO] = None

    category: ClassVar[BlockCategory] = BlockCategory.DECISION

    def validate(self) -> None:
        BaseBlock.validate(self)
        if self.condition is not None and not isinstance(self.condition, ConditionDTO):
            raise TypeError("condition must be a ConditionDTO instance.")

    def to_json(self) -> Dict[str, Any]:
        payload = BaseBlock.to_json(self)
        if self.condition is not None:
            payload["condition"] = self.condition.to_json()
        return payload


class IfElseDecisionBlock(BaseDecisionBlock):
    block_type = DecisionBlockType.IF_ELSE.value
    block_name = "If Else Decision"

    def predefined_config(self) -> Dict[str, Any]:
        return {
            "operator_type": "conditional_branch",
            "branches": ["true", "false"],
            "condition_schema": {
                "left_operand": "string",
                "operator": "string",
                "right_operand": "string",
            },
        }


class OrDecisionBlock(BaseDecisionBlock):
    block_type = DecisionBlockType.OR.value
    block_name = "Or Decision"

    def predefined_config(self) -> Dict[str, Any]:
        return {
            "operator_type": "logical_or",
            "min_inputs": 2,
            "output_when": "any_input_matches",
        }


class AndDecisionBlock(BaseDecisionBlock):
    block_type = DecisionBlockType.AND.value
    block_name = "And Decision"

    def predefined_config(self) -> Dict[str, Any]:
        return {
            "operator_type": "logical_and",
            "min_inputs": 2,
            "output_when": "all_inputs_match",
        }
