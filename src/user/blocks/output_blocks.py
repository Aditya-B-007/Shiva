from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from src.user.blocks.base import BaseBlock
from src.user.constants import BlockCategory, OutputBlockType, OutputFormat


@dataclass(slots=True)
class ShivaOutputBlock(BaseBlock):
    output_format: OutputFormat = OutputFormat.TEXT

    category = BlockCategory.OUTPUT
    block_type = OutputBlockType.SHIVA_OUTPUT.value
    block_name = "Shiva Output"

    def validate(self) -> None:
        BaseBlock.validate(self)
        if isinstance(self.output_format, str):
            self.output_format = OutputFormat(self.output_format)
        if self.output_format not in (OutputFormat.TEXT, OutputFormat.VOICE):
            raise ValueError("output_format must be either text or voice.")

    def predefined_config(self) -> Dict[str, Any]:
        return {
            "producer": "shiva",
            "format": self.output_format.value,
            "supported_formats": [OutputFormat.TEXT.value, OutputFormat.VOICE.value],
        }
