from dataclasses import dataclass
from typing import Optional

@dataclass
class WriteRequest:
    entry_id: str
    action: str  # "SAVE", "UPDATE", "DELETE", "CLEAR"
    content: Optional[str] = None
    confidence: Optional[float] = None
