from __future__ import annotations

from typing import Any, Dict

from src.user.blocks.base import BaseBlock
from src.user.constants import BlockCategory, InputBlockType


class CameraInputBlock(BaseBlock):
    category = BlockCategory.INPUT
    block_type = InputBlockType.CAMERA.value
    block_name = "Camera Input"

    def predefined_config(self) -> Dict[str, Any]:
        return {
            "device": "camera",
            "capture_mode": "image",
            "required_permission": "camera",
            "accepted_payloads": ["image_bytes", "image_path"],
        }


class MicrophoneInputBlock(BaseBlock):
    category = BlockCategory.INPUT
    block_type = InputBlockType.MICROPHONE.value
    block_name = "Microphone Input"

    def predefined_config(self) -> Dict[str, Any]:
        return {
            "device": "microphone",
            "capture_mode": "audio",
            "required_permission": "microphone",
            "accepted_payloads": ["audio_bytes", "transcript"],
        }


class NetworkInputBlock(BaseBlock):
    category = BlockCategory.INPUT
    block_type = InputBlockType.NETWORK.value
    block_name = "Network Input"

    def predefined_config(self) -> Dict[str, Any]:
        return {
            "device": "network",
            "capture_mode": "request",
            "required_permission": "network",
            "accepted_payloads": ["url", "headers", "body", "method"],
        }


class PromptInputBlock(BaseBlock):
    category = BlockCategory.INPUT
    block_type = InputBlockType.PROMPT.value
    block_name = "Prompt Input"

    def predefined_config(self) -> Dict[str, Any]:
        return {
            "device": "user_prompt",
            "capture_mode": "text",
            "required_permission": None,
            "accepted_payloads": ["prompt_text"],
        }
