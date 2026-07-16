from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class BlockCategory(str, Enum):
    INPUT = "input"
    DECISION = "decision"
    OUTPUT = "output"


class InputBlockType(str, Enum):
    CAMERA = "camera"
    MICROPHONE = "microphone"
    NETWORK = "network"
    PROMPT = "prompt"


class DecisionBlockType(str, Enum):
    IF_ELSE = "if_else"
    OR = "or"
    AND = "and"


class OutputBlockType(str, Enum):
    SHIVA_OUTPUT = "shiva_output"


@dataclass(frozen=True, slots=True)
class BlockFieldSchema:
    name: str
    label: str
    field_type: str = "text"
    default: Any = None
    options: List[str] = field(default_factory=list)
    required: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "field_type": self.field_type,
            "default": self.default,
            "options": list(self.options),
            "required": self.required,
        }


@dataclass(frozen=True, slots=True)
class BlockSchema:
    block_type: str
    name: str
    category: BlockCategory
    default_arguments: Dict[str, Any] = field(default_factory=dict)
    fields: List[BlockFieldSchema] = field(default_factory=list)
    condition: Dict[str, Any] = field(default_factory=dict)
    execution_device: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "block_type": self.block_type,
            "name": self.name,
            "category": self.category.value,
            "default_arguments": dict(self.default_arguments),
            "fields": [field_schema.to_json() for field_schema in self.fields],
            "condition": dict(self.condition),
            "execution_device": self.execution_device,
        }


def block_schemas() -> List[BlockSchema]:
    return [
        BlockSchema(
            block_type=InputBlockType.CAMERA.value,
            name="Camera Input",
            category=BlockCategory.INPUT,
            execution_device="camera",
            default_arguments={
                "device": "camera",
                "capture_mode": "image",
                "required_permission": "camera",
            },
            fields=[
                BlockFieldSchema("device", "Device Key", default="camera", required=True),
                BlockFieldSchema("capture_mode", "Capture Mode", default="image", required=True),
                BlockFieldSchema("required_permission", "Required Permission", default="camera"),
            ],
        ),
        BlockSchema(
            block_type=InputBlockType.MICROPHONE.value,
            name="Microphone Input",
            category=BlockCategory.INPUT,
            execution_device="microphone",
            default_arguments={
                "device": "microphone",
                "capture_mode": "audio",
                "required_permission": "microphone",
            },
            fields=[
                BlockFieldSchema("device", "Device Key", default="microphone", required=True),
                BlockFieldSchema("capture_mode", "Capture Mode", default="audio", required=True),
                BlockFieldSchema("required_permission", "Required Permission", default="microphone"),
            ],
        ),
        BlockSchema(
            block_type=InputBlockType.NETWORK.value,
            name="Network Input",
            category=BlockCategory.INPUT,
            execution_device="network",
            default_arguments={
                "device": "network",
                "capture_mode": "request",
                "required_permission": "network",
                "url": "https://example.com",
                "method": "GET",
            },
            fields=[
                BlockFieldSchema("device", "Device Key", default="network", required=True),
                BlockFieldSchema("capture_mode", "Capture Mode", default="request", required=True),
                BlockFieldSchema("required_permission", "Required Permission", default="network"),
                BlockFieldSchema("url", "Target Request URL", default="https://example.com"),
                BlockFieldSchema("method", "HTTP Method", default="GET"),
            ],
        ),
        BlockSchema(
            block_type=InputBlockType.PROMPT.value,
            name="Prompt Input",
            category=BlockCategory.INPUT,
            execution_device="user_prompt",
            default_arguments={
                "device": "user_prompt",
                "capture_mode": "text",
            },
            fields=[
                BlockFieldSchema("device", "Device Key", default="user_prompt", required=True),
                BlockFieldSchema("capture_mode", "Capture Mode", default="text", required=True),
                BlockFieldSchema("required_permission", "Required Permission", default=None),
            ],
        ),
        BlockSchema(
            block_type=DecisionBlockType.IF_ELSE.value,
            name="If Else Decision",
            category=BlockCategory.DECISION,
            default_arguments={"operator_type": "conditional_branch"},
            condition={
                "left_operand": "status",
                "operator": "==",
                "right_operand": "online",
            },
            fields=[
                BlockFieldSchema("left_operand", "Left Operand", default="status"),
                BlockFieldSchema("operator", "Comparison Operator", default="=="),
                BlockFieldSchema("right_operand", "Right Operand", default="online"),
            ],
        ),
        BlockSchema(
            block_type=DecisionBlockType.OR.value,
            name="Or Decision",
            category=BlockCategory.DECISION,
            default_arguments={"operator_type": "logical_or", "min_inputs": 2},
            fields=[
                BlockFieldSchema("min_inputs", "Min Core Inputs", field_type="number", default=2),
            ],
        ),
        BlockSchema(
            block_type=DecisionBlockType.AND.value,
            name="And Decision",
            category=BlockCategory.DECISION,
            default_arguments={"operator_type": "logical_and", "min_inputs": 2},
            fields=[
                BlockFieldSchema("min_inputs", "Min Core Inputs", field_type="number", default=2),
            ],
        ),
        BlockSchema(
            block_type=OutputBlockType.SHIVA_OUTPUT.value,
            name="Shiva Output",
            category=BlockCategory.OUTPUT,
            default_arguments={"output_format": "text"},
            fields=[
                BlockFieldSchema(
                    "output_format",
                    "Output Format",
                    field_type="select",
                    default="text",
                    options=["text", "voice"],
                    required=True,
                ),
            ],
        ),
    ]
