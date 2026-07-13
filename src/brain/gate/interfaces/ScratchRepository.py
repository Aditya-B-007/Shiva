from abc import ABC, abstractmethod
from typing import List, Optional
from ..models.ScratchEntry import ScratchEntry

class ScratchRepository(ABC):
    """Interface for temporary thoughts (ScratchPad) repository."""
    
    @abstractmethod
    def save(self, entry: ScratchEntry) -> None:
        """Save a new temporary thought."""
        pass

    @abstractmethod
    def load(self, entry_id: str) -> Optional[ScratchEntry]:
        """Load a temporary thought by its ID."""
        pass

    @abstractmethod
    def update(self, entry: ScratchEntry) -> None:
        """Update an existing temporary thought."""
        pass

    @abstractmethod
    def delete(self, entry_id: str) -> None:
        """Delete a temporary thought by its ID."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all temporary thoughts."""
        pass

    @abstractmethod
    def load_all(self) -> List[ScratchEntry]:
        """Load all temporary thoughts."""
        pass
