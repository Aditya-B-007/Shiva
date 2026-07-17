from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Iterable, Mapping

from src.transferDTO import ObservationKind, PerceptionObservationDTO, PerceptionBundleDTO, PerceptionCaptureRequestDTO

class PerceptionObservationFactory:
    def from_capture(self, device: str, payload: Any) -> PerceptionObservationDTO:
        kind = self._classify(device, payload)
        payload_size = self._payload_size(payload)
        return PerceptionObservationDTO(
            device=device,
            kind=kind,
            summary=self._summarize(device, kind, payload, payload_size),
            payload=payload,
            payload_size=payload_size,
            metadata=self._metadata(payload),
        )

    def from_error(self, device: str, error: Exception) -> PerceptionObservationDTO:
        return PerceptionObservationDTO(
            device=device,
            kind=ObservationKind.ERROR,
            summary=f"{type(error).__name__}: {error}",
            payload=None,
            payload_size=0,
            metadata={"error_type": type(error).__name__},
        )

    def _classify(self, device: str, payload: Any) -> ObservationKind:
        device_name = device.lower()
        if isinstance(payload, str):
            return ObservationKind.TEXT
        if isinstance(payload, bytes):
            return ObservationKind.BINARY_BYTES
        if isinstance(payload, Mapping):
            return ObservationKind.STRUCTURED
        return ObservationKind.UNKNOWN

    def _payload_size(self, payload: Any) -> int:
        if payload is None:
            return 0
        if isinstance(payload, (bytes, str, list, tuple, set, dict)):
            return len(payload)
        return 1

    def _metadata(self, payload: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {"payload_type": type(payload).__name__}
        if isinstance(payload, Mapping):
            metadata["keys"] = list(payload.keys())
        return metadata

    def _summarize(
        self,
        device: str,
        kind: ObservationKind,
        payload: Any,
        payload_size: int,
    ) -> str:
        if kind == ObservationKind.TEXT:
            text = str(payload).replace("\n", " ").strip()
            return text[:240] if text else f"{device} returned empty text."
        if kind == ObservationKind.BINARY_BYTES:
            return f"{device} captured binary payload ({payload_size} bytes)."
        if kind == ObservationKind.STRUCTURED and isinstance(payload, Mapping):
            parts = [f"{key}={self._compact_value(value)}" for key, value in payload.items()]
            return "; ".join(parts)[:360]
        return f"{device} captured {type(payload).__name__} payload."

    def _compact_value(self, value: Any) -> str:
        if isinstance(value, bytes):
            return f"<bytes:{len(value)}>"
        text = str(value).replace("\n", " ").strip()
        return text[:80]

class PerceptionPromptFormatter:
    def format_bundle(self, bundle: PerceptionBundleDTO) -> str:
        lines = [f"Query: {bundle.query}"]
        if not bundle.observations:
            lines.append("Observations: none")
            return "\n".join(lines)

        lines.append("Observations:")
        lines.extend(self.format_observation(observation) for observation in bundle.observations)
        return "\n".join(lines)

    def format_observation(self, observation: PerceptionObservationDTO) -> str:
        return (
            f"- {observation.device}: kind={observation.kind.value}; "
            f"summary={observation.summary}; size={observation.payload_size}"
        )

    def format_observation_names(self, observations: Iterable[PerceptionObservationDTO]) -> List[str]:
        return [observation.device for observation in observations]
