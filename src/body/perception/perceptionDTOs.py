from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class ObservationKind(str, Enum):
    TEXT = "text"
    IMAGE_BYTES = "image_bytes"
    AUDIO_BYTES = "audio_bytes"
    BINARY_BYTES = "binary_bytes"
    STRUCTURED = "structured"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class PerceptionObservationDTO:
    device: str
    kind: ObservationKind
    summary: str
    payload: Any = None
    payload_size: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    captured_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class PerceptionBundleDTO:
    query: str
    observations: List[PerceptionObservationDTO] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def has_observations(self) -> bool:
        return bool(self.observations)

    def observation_names(self) -> List[str]:
        return [observation.device for observation in self.observations]

    def errors(self) -> List[PerceptionObservationDTO]:
        return [
            observation
            for observation in self.observations
            if observation.kind == ObservationKind.ERROR
        ]


@dataclass(frozen=True, slots=True)
class PerceptionCaptureRequestDTO:
    device: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    request_id: Optional[str] = None
