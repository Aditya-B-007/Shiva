from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from src.transferDTO import EmotionDTO

@dataclass
class Memory:
    """Represents a permanent cognitive memory after promotion from ScratchPad."""
    id: str
    content: str
    confidence: float
    created_at: datetime
    promoted_at: datetime
    emotion: Optional[EmotionDTO] = None
