from dataclasses import dataclass
from typing import Optional
from src.brain.emotionalHandlerAndStore.emotionalContract import EmotionDTO

@dataclass
class WriteRequest:
    """Represents a write request from a reasoning node to the cognitive gate system."""
    entry_id: str
    action: str  # "SAVE", "UPDATE", "DELETE", "CLEAR"
    content: Optional[str] = None
    confidence: Optional[float] = None
    emotion: Optional[EmotionDTO] = None
