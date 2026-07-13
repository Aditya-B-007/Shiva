from typing import List, Optional
from .models.WriteRequest import WriteRequest
from .models.ScratchEntry import ScratchEntry
from .models.Memory import Memory
from .ScratchPadGate import ScratchPadGate
from .MemoryGate import MemoryGate
from ..infrastructure.queue.QueueManager import QueueManager

class Gate:

    def __init__(
        self,
        queue_manager: QueueManager,
        scratch_gate: ScratchPadGate,
        memory_gate: MemoryGate
    ) -> None:
        self.queue_manager = queue_manager
        self.scratch_gate = scratch_gate
        self.memory_gate = memory_gate

    def submit(self, request: WriteRequest) -> None:
        self.queue_manager.enqueue(request)

    def read_scratch(self, entry_id: str) -> Optional[ScratchEntry]:
        return self.scratch_gate.load(entry_id)

    def read_memory(self, memory_id: str) -> Optional[Memory]:
        return self.memory_gate.load(memory_id)

    def list_scratch(self) -> List[ScratchEntry]:
        return self.scratch_gate.load_all()

    def list_memory(self) -> List[Memory]:
        return self.memory_gate.load_all()
