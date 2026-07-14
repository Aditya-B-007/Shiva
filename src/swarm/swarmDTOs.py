from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass(slots=True)
class MothershipResponseDTO:
    decision: Optional[str]
    confidence: float
    goal_reached: bool
    observations_used: List[str] = field(default_factory=list)
    reasoning_summary: str = ""
    cycles_used: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)
