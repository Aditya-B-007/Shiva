from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict

from src.user.constants import BlockCategory


@dataclass(slots=True)
class BaseBlock(ABC):
    block_id: str
    user_values: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    category: ClassVar[BlockCategory]
    block_type: ClassVar[str]
    block_name: ClassVar[str]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.block_id or not self.block_id.strip():
            raise ValueError("block_id is required.")
        if not isinstance(self.user_values, dict):
            raise TypeError("user_values must be a dictionary.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

    @abstractmethod
    def predefined_config(self) -> Dict[str, Any]:
        """Return the immutable block-specific JSON structure expected by Shiva."""

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.block_id,
            "category": self.category.value,
            "type": self.block_type,
            "name": self.block_name,
            "config": self.predefined_config(),
            "user_values": dict(self.user_values),
            "metadata": dict(self.metadata),
        }

    def get_raw_prompt_fragment(self) -> Dict[str, Any]:
        return self.to_json()
