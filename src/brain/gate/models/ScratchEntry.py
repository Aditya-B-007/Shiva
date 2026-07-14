from dataclasses import dataclass

@dataclass
class ScratchEntry:
    id: str
    content: str
    confidence: float
    created_at: float
    updated_at: float
