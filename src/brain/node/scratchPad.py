from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.transferDTO import ThoughtDTO, ReasoningContextDTO


@dataclass
class ScratchPad:

    perception: Optional[Any] = None

    retrieved_memories: List[Any] = field(default_factory=list)

    emotional_state: Optional[Any] = None

    thoughts: List[ThoughtDTO] = field(default_factory=list)

    intermediate_results: List[Any] = field(default_factory=list)

    hypotheses: List[Any] = field(default_factory=list)

    context: Dict[str, Any] = field(default_factory=dict)

    decision: Optional[Any] = None

    confidence: float = 0.0

    def initialize(
        self,
        perception: Any,
        memory: List[Any],
        emotion: Any
    ) -> None:
        self.perception = perception
        self.retrieved_memories = memory
        self.emotional_state = emotion

    def append_thought(self, thought: Any) -> None:
        self.thoughts.append(thought)

    def add_hypothesis(self, hypothesis: Any) -> None:
        self.hypotheses.append(hypothesis)

    def add_intermediate_result(self, result: Any) -> None:
        self.intermediate_results.append(result)

    def update_context(
        self,
        key: str,
        value: Any
    ) -> None:
        self.context[key] = value

    def current_context(self) -> ReasoningContextDTO:
        return ReasoningContextDTO(
            perception=self.perception,
            memories=self.retrieved_memories,
            emotion=self.emotional_state,
            thoughts=list(self.thoughts),
            hypotheses=list(self.hypotheses),
            context=dict(self.context),
        )
    
    def set_decision(
        self,
        decision: Any,
        confidence: float
    ) -> None:
        self.decision = decision
        self.confidence = confidence

    def final_state(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "thoughts": self.thoughts,
            "memory": self.retrieved_memories,
            "emotion": self.emotional_state,
            "context": self.context,
        }

    def clear(self) -> None:
        self.perception = None
        self.retrieved_memories.clear()
        self.emotional_state = None
        self.thoughts.clear()
        self.hypotheses.clear()
        self.intermediate_results.clear()
        self.context.clear()
        self.decision = None
        self.confidence = 0.0
