from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Optional
from datetime import datetime

@dataclass(slots=True)
class ThoughtDTO:
    raw_text: str
    thought_body: str
    critique: str
    confidence: float
    parsed_decision: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass(slots=True)
class ReasoningContextDTO:
    perception: Any
    memories: List[Any] = field(default_factory=list)
    emotion: Optional[Any] = None
    thoughts: List[ThoughtDTO] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass(slots=True)
class NodeReasoningResultDTO:
    decision: Optional[str]
    confidence: float
    thought_history: List[ThoughtDTO]
    iterations_used: int
    max_iterations: int
    goal_reached: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
