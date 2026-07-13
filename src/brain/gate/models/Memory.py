from dataclasses import dataclass
from typing import Optional
from src.brain.emotionalHandlerAndStore.emotionalContract import EmotionDTO

@dataclass
class Memory:
    """Represents a permanent cognitive memory after promotion from ScratchPad."""
    id: str
    content: str
    confidence: float
    created_at: float
    promoted_at: float
    emotion: Optional[EmotionDTO] = None
