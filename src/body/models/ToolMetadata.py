from dataclasses import dataclass
from typing import Any, Dict

@dataclass(slots=True)
class ToolMetadata:
    """Represents a clean metadata specification of a tool for the Decoder to understand."""
    name: str
    description: str
    parameter_definitions: Dict[str, Any]
