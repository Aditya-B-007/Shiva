from typing import List, Optional
from .interfaces.ScratchRepository import ScratchRepository
from .models.ScratchEntry import ScratchEntry

class ScratchPadGate:

    def __init__(self, repository: ScratchRepository) -> None:
        self.repository = repository

    def save(self, entry: ScratchEntry) -> None:
        self.repository.save(entry)

    def load(self, entry_id: str) -> Optional[ScratchEntry]:
        return self.repository.load(entry_id)

    def update(self, entry: ScratchEntry) -> None:
        self.repository.update(entry)

    def delete(self, entry_id: str) -> None:
        self.repository.delete(entry_id)

    def clear(self) -> None:
        self.repository.clear()

    def load_all(self) -> List[ScratchEntry]:
        return self.repository.load_all()
