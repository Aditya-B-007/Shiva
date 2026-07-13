from abc import ABC, abstractmethod
from typing import List, Optional
from ..models.Memory import Memory

class MemoryRepository(ABC):
    """Interface for permanent memories repository."""

    @abstractmethod
    def save(self, memory: Memory) -> None:
        """Save a promoted permanent memory."""
        pass

    @abstractmethod
    def load(self, memory_id: str) -> Optional[Memory]:
        """Load a permanent memory by its ID."""
        pass

    @abstractmethod
    def update(self, memory: Memory) -> None:
        """Update an existing permanent memory."""
        pass

    @abstractmethod
    def delete(self, memory_id: str) -> None:
        """Delete a permanent memory by its ID."""
        pass

    @abstractmethod
    def load_all(self) -> List[Memory]:
        """Load all permanent memories."""
        pass
