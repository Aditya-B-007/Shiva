from dataclasses import dataclass
from typing import Optional
from src.brain.emotionalHandlerAndStore.emotionalContract import EmotionDTO

@dataclass
class ScratchEntry:
    """Represents a temporary thought entry in the ScratchPad memory."""
    id: str
    content: str
    confidence: float
    created_at: float
    updated_at: float
    emotion: Optional[EmotionDTO] = None
