from dataclasses import dataclass
from datetime import datetime

@dataclass
class ScratchEntry:
    id: str
    content: str
    confidence: float
    created_at: datetime
    updated_at: datetime
