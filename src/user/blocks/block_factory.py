from __future__ import annotations

from typing import Any, Dict, Type

from src.user.blocks.base import BaseBlock
from src.user.blocks.decision_blocks import (
    AndDecisionBlock,
    IfElseDecisionBlock,
    OrDecisionBlock,
)
from src.user.blocks.input_blocks import (
    CameraInputBlock,
    MicrophoneInputBlock,
    NetworkInputBlock,
    PromptInputBlock,
)
from src.user.blocks.output_blocks import ShivaOutputBlock
from src.user.constants import DecisionBlockType, InputBlockType, OutputBlockType


class BlockFactory:
    _registry: Dict[str, Type[BaseBlock]] = {
        InputBlockType.CAMERA.value: CameraInputBlock,
        InputBlockType.MICROPHONE.value: MicrophoneInputBlock,
        InputBlockType.NETWORK.value: NetworkInputBlock,
        InputBlockType.PROMPT.value: PromptInputBlock,
        DecisionBlockType.IF_ELSE.value: IfElseDecisionBlock,
        DecisionBlockType.OR.value: OrDecisionBlock,
        DecisionBlockType.AND.value: AndDecisionBlock,
        OutputBlockType.SHIVA_OUTPUT.value: ShivaOutputBlock,
    }

    @classmethod
    def create_block(
        cls,
        block_type: str,
        block_id: str,
        **kwargs: Any,
    ) -> BaseBlock:
        block_class = cls._registry.get(block_type)
        if block_class is None:
            supported = ", ".join(sorted(cls._registry))
            raise ValueError(f"Unsupported block_type '{block_type}'. Supported: {supported}")
        return block_class(block_id=block_id, **kwargs)

    @classmethod
    def register_block(cls, block_type: str, block_class: Type[BaseBlock]) -> None:
        if not issubclass(block_class, BaseBlock):
            raise TypeError("block_class must inherit from BaseBlock.")
        cls._registry[block_type] = block_class

    @classmethod
    def supported_block_types(cls) -> Dict[str, Type[BaseBlock]]:
        return dict(cls._registry)
