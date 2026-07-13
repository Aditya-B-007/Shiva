from dataclasses import dataclass
from typing import Any, Optional

@dataclass(slots=True)
class ToolResult:
    """Encapsulates the output and execution status of a tool."""
    success: bool
    output: Any
    error: Optional[str] = None
    timestamp: float = 0.0
