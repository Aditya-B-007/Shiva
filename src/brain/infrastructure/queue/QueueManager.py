import queue
from typing import Optional
from ...gate.models.WriteRequest import WriteRequest

class QueueManager:
    """Owns the thread-safe queue containing write requests."""

    def __init__(self) -> None:
        self._queue: queue.Queue[WriteRequest] = queue.Queue()

    def enqueue(self, request: WriteRequest) -> None:
        """Enqueue a write request."""
        self._queue.put(request)

    def dequeue(self, block: bool = True, timeout: Optional[float] = None) -> WriteRequest:
        """Dequeue a write request. Blocks by default."""
        return self._queue.get(block=block, timeout=timeout)

    def size(self) -> int:
        """Get the current size of the queue."""
        return self._queue.qsize()
