import threading
from contextlib import contextmanager
from typing import Any, Optional, Dict, List
from src.brain.node.nodeDTOs import BrainErrorDTO, NodeReasoningResultDTO, ThoughtDTO, ReasoningContextDTO

class ReasoningScheduler:
    def __init__(self, decoder: Any):
        self.decoder = decoder
        self._lock = threading.Lock()

    @contextmanager
    def acquire_decoder(self):
        self._lock.acquire()
        try:
            yield self.decoder
        finally:
            self._lock.release()


class nodeProcessingEngine:
    def __init__(
        self,
        memory_engine: Any,
        emotion_handler: Any,
        scratchpad: Any,
        chain_of_thought: Any,
        reasoning_scheduler: ReasoningScheduler
    ):
        """
        Local cognitive orchestrator of a single Shiva node.
        """
        self._memory = memory_engine
        self._emotion = emotion_handler
        self._scratchpad = scratchpad
        self._chain = chain_of_thought
        self._reasoning_scheduler = reasoning_scheduler

    def process(
        self,
        perception: Any,
        seed_thoughts: List[ThoughtDTO] = None,
        memories: List[Any] = None,
        decoder_kwargs: Optional[Dict[str, Any]] = None,
    ) -> NodeReasoningResultDTO:
        errors: List[BrainErrorDTO] = []
        self._chain.reset()
        self._scratchpad.clear()
        if memories is None:
            try:
                raw_memories = self._memory.retrieve(perception)
                memories = getattr(raw_memories, "memories", raw_memories)
                if not isinstance(memories, list) and isinstance(memories, tuple):
                    memories = list(memories)
            except Exception as exc:
                memories = []
                errors.append(BrainErrorDTO("memory", str(exc), recoverable=True))
        try:
            if hasattr(self._emotion, "perceive_event"):
                emotion = self._emotion.perceive_event(perception)
            elif hasattr(self._emotion, "process"):
                emotion = self._emotion.process(perception, memories)
            elif hasattr(self._emotion, "current_emotion"):
                emotion = self._emotion.current_emotion()
            else:
                emotion = getattr(self._emotion, "emotion", None)
        except Exception as exc:
            emotion = None
            errors.append(BrainErrorDTO("emotion", str(exc), recoverable=True))

        self._scratchpad.initialize(
            perception=perception,
            memory=memories,
            emotion=emotion
        )
        for thought in seed_thoughts or []:
            self._scratchpad.append_thought(thought)
        while self._chain.should_continue():
            # Request one decoder time slice
            try:
                with self._reasoning_scheduler.acquire_decoder() as decoder:
                    # Use generateDecision method to output a structured ThoughtDTO
                    thought_dto: ThoughtDTO = decoder.generateDecision(
                        self._scratchpad.current_context(),
                        **(decoder_kwargs or {}),
                    )
            except Exception as exc:
                errors.append(BrainErrorDTO("decoder", str(exc), recoverable=False))
                self._chain.terminate()
                break

            # Append the thought and update loop state
            self._scratchpad.append_thought(thought_dto)
            self._chain.update(
                thought=thought_dto,
                scratchpad=self._scratchpad
            )

        # 3. Produce and return standardized result DTO
        return NodeReasoningResultDTO(
            decision=self._scratchpad.decision,
            confidence=self._scratchpad.confidence,
            thought_history=list(self._scratchpad.thoughts),
            iterations_used=self._chain.iterations_used(),
            max_iterations=self._chain.max_iterations,
            goal_reached=self._chain.goal_reached,
            errors=errors,
        )

NodeProcessingEngine = nodeProcessingEngine
