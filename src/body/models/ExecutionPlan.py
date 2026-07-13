from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass(slots=True)
class ExecutionPlan:
    """Represents the Decoder's decision specifying which tool to execute and its arguments."""
    tool_name: str
    arguments: Dict[str, Any]
    reasoning: Optional[str] = None
    decoder_confidence: Optional[float] = None
