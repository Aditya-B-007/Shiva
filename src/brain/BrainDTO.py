from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class BrainErrorDTO:
    component: str
    message: str
    recoverable: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class ThoughtParseDiagnosticsDTO:
    parse_success: bool = True
    missing_fields: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ThoughtDTO:
    raw_text: str
    thought_body: str
    critique: str
    confidence: float
    parsed_decision: Optional[str] = None
    parse_diagnostics: ThoughtParseDiagnosticsDTO = field(default_factory=ThoughtParseDiagnosticsDTO)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class ReasoningContextDTO:
    perception: Any
    memories: List[Any] = field(default_factory=list)
    emotion: Optional[Any] = None
    thoughts: List[ThoughtDTO] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class NodeReasoningResultDTO:
    decision: Optional[str]
    confidence: float
    thought_history: List[ThoughtDTO]
    iterations_used: int
    max_iterations: int
    goal_reached: bool
    errors: List[BrainErrorDTO] = field(default_factory=list)
    timed_out: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class ColumnResultDTO:
    column_id: int
    role: str
    result: Optional[NodeReasoningResultDTO] = None
    error: Optional[BrainErrorDTO] = None
    duration_ms: float = 0.0


@dataclass(slots=True)
class MothershipResponseDTO:
    decision: Optional[str]
    confidence: float
    goal_reached: bool
    observations_used: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    cycles_used: int = 0
    errors: List[BrainErrorDTO] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(slots=True)
class BrainHealthDTO:
    status: str
    cognitive_core_online: bool
    decoder_online: bool
    memory_online: bool
    homeostasis_online: bool
    errors: List[BrainErrorDTO] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
