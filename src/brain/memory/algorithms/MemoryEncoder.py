from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
from typing import Any, Mapping

from ..graph.MemoryNode import MemoryModality, MemoryNode, MemoryType, clamp_unit


class MemoryEncoder:
    def encode(
        self,
        perception: Any,
        emotion: Any = None,
        homeostasis: Any = None,
        context: Mapping[str, Any] | None = None,
    ) -> MemoryNode:
        raw_content = self._raw_content(perception)
        summary = self._summary(raw_content)
        modality = self._modality(perception, raw_content)
        context_payload = dict(context or {})
        return MemoryNode(
            raw_content=raw_content,
            summary=summary,
            modality=modality,
            semantic_type=self._semantic_type(context_payload),
            activation=self._activation(emotion, homeostasis, context_payload),
            strength=self._strength(emotion, context_payload),
            emotional_salience=self._emotional_salience(emotion),
            identity_relevance=self._identity_relevance(context_payload),
            context_signature=self._context_signature(context_payload),
        )

    def _raw_content(self, perception: Any) -> Any:
        if hasattr(perception, "payload"):
            return getattr(perception, "payload")
        if hasattr(perception, "text") and getattr(perception, "text"):
            return getattr(perception, "text")
        return perception

    def _summary(self, raw_content: Any) -> str:
        text = str(raw_content)
        return text if len(text) <= 160 else f"{text[:157]}..."

    def _modality(self, perception: Any, raw_content: Any) -> MemoryModality:
        if hasattr(perception, "sensor_data") and getattr(perception, "sensor_data"):
            return MemoryModality.SENSOR
        if isinstance(raw_content, Mapping) and raw_content.get("action"):
            return MemoryModality.ACTION
        return MemoryModality.TEXT

    def _semantic_type(self, context: Mapping[str, Any]) -> MemoryType:
        raw_type = context.get("semantic_type") or context.get("memory_type")
        if raw_type:
            return MemoryType(str(raw_type))
        return MemoryType.EPISODIC

    def _activation(self, emotion: Any, homeostasis: Any, context: Mapping[str, Any]) -> float:
        importance = float(context.get("importance", 0.5))
        intensity = self._float_attr(emotion, "emotional_intensity", 0.0)
        focus = self._float_attr(homeostasis, "focus", 0.5)
        return clamp_unit((importance + intensity + focus) / 3.0)

    def _strength(self, emotion: Any, context: Mapping[str, Any]) -> float:
        importance = float(context.get("importance", 0.5))
        intensity = self._float_attr(emotion, "emotional_intensity", 0.0)
        return clamp_unit(0.35 + importance * 0.35 + intensity * 0.30)

    def _emotional_salience(self, emotion: Any) -> float:
        return clamp_unit(self._float_attr(emotion, "emotional_intensity", 0.0))

    def _identity_relevance(self, context: Mapping[str, Any]) -> float:
        return clamp_unit(float(context.get("identity_relevance", context.get("goal_relevance", 0.0))))

    def _context_signature(self, context: Mapping[str, Any]) -> str:
        normalized = self._normalize(context)
        return sha256(repr(sorted(normalized.items())).encode("utf-8")).hexdigest()

    def _normalize(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value) # type: ignore
        if isinstance(value, Mapping):
            return {str(key): self._normalize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._normalize(item) for item in value]
        return value

    def _float_attr(self, value: Any, attr: str, default: float) -> float:
        if value is None:
            return default
        if hasattr(value, attr):
            try:
                return float(getattr(value, attr))
            except (TypeError, ValueError):
                return default
        if isinstance(value, Mapping) and attr in value:
            try:
                return float(value[attr])
            except (TypeError, ValueError):
                return default
        return default
