from typing import List, Optional
from .interfaces.MemoryRepository import MemoryRepository
from .models.Memory import Memory

class MemoryGate:

    def __init__(self, repository: MemoryRepository) -> None:
        self.repository = repository

    def save(self, memory: Memory) -> None:
        """Save promoted memories."""
        self.repository.save(memory)

    def load(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memories."""
        return self.repository.load(memory_id)

    def update(self, memory: Memory) -> None:
        """Update memories."""
        self.repository.update(memory)

    def delete(self, memory_id: str) -> None:
        """Delete memories."""
        self.repository.delete(memory_id)

    def load_all(self) -> List[Memory]:
        """Load all memories."""
        return self.repository.load_all()
