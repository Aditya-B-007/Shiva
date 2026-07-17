import time
from datetime import datetime
import logging
import threading
import queue
from typing import Optional
from ...gate.models.WriteRequest import WriteRequest
from ...gate.models.ScratchEntry import ScratchEntry
from ...gate.models.Memory import Memory
from ...gate.ScratchPadGate import ScratchPadGate
from ...gate.MemoryGate import MemoryGate
from ...gate.services.PromotionPolicy import PromotionPolicy
from ...gate.services.TransactionService import TransactionService
from .QueueManager import QueueManager

logger = logging.getLogger(__name__)

class GateWorker:
    """Consumes requests from QueueManager continuously and coordinates write execution flows.
    
    Guarantees deterministic writes and removes race conditions as the single writer.
    """

    def __init__(
        self,
        queue_manager: QueueManager,
        transaction_service: TransactionService,
        scratch_gate: ScratchPadGate,
        memory_gate: MemoryGate,
        promotion_policy: PromotionPolicy
    ) -> None:
        self.queue_manager = queue_manager
        self.transaction_service = transaction_service
        self.scratch_gate = scratch_gate
        self.memory_gate = memory_gate
        self.promotion_policy = promotion_policy
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the background worker thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name="GateWorkerThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background worker thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _run_loop(self) -> None:
        while self._running:
            try:
                # Use a small timeout so the loop can react to self._running changes
                request = self.queue_manager.dequeue(block=True, timeout=1.0)
            except queue.Empty:
                continue

            try:
                # Wrap all operations in a transaction via the TransactionService
                self.transaction_service.execute_transaction(
                    request,
                    lambda: self._process_request(request)
                )
            except Exception as e:
                logger.error(f"Error processing write request {request}: {e}", exc_info=True)

    def _process_request(self, request: WriteRequest) -> None:
        if request.action == "SAVE":
            entry = ScratchEntry(
                id=request.entry_id,
                content=request.content or "",
                confidence=request.confidence or 0.0,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                emotion=request.emotion
            )
            self.scratch_gate.save(entry)
            self._check_and_promote(entry)
            
        elif request.action == "UPDATE":
            existing = self.scratch_gate.load(request.entry_id)
            created_at = existing.created_at if existing else datetime.utcnow()
            entry = ScratchEntry(
                id=request.entry_id,
                content=request.content if request.content is not None else (existing.content if existing else ""),
                confidence=request.confidence if request.confidence is not None else (existing.confidence if existing else 0.0),
                created_at=created_at,
                updated_at=datetime.utcnow(),
                emotion=request.emotion if request.emotion is not None else (existing.emotion if existing else None)
            )
            self.scratch_gate.update(entry)
            self._check_and_promote(entry)
            
        elif request.action == "DELETE":
            self.scratch_gate.delete(request.entry_id)
            
        elif request.action == "CLEAR":
            self.scratch_gate.clear()

    def _check_and_promote(self, entry: ScratchEntry) -> None:
        """Evaluates promotion and shifts promoted entries into permanent MemoryGate."""
        if self.promotion_policy.should_promote(entry):
            memory = Memory(
                id=entry.id,
                content=entry.content,
                confidence=entry.confidence,
                created_at=entry.created_at,
                promoted_at=datetime.utcnow(),
                emotion=entry.emotion
            )
            self.memory_gate.save(memory)
            self.scratch_gate.delete(entry.id)
            logger.info(f"Promoted scratchpad entry {entry.id} to memory.")
