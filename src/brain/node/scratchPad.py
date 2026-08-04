from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from src.transferDTO import ThoughtDTO, ReasoningContextDTO, ThoughtStepDTO
from src.brain.transformer.thought_parser import ThoughtParser


@dataclass
class ScratchPad:
    perception: Optional[Any] = None
    retrieved_memories: List[Any] = field(default_factory=list)
    emotional_state: Optional[Any] = None
    thoughts: List[ThoughtDTO] = field(default_factory=list)
    steps: List[ThoughtStepDTO] = field(default_factory=list)
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
        """
        Appends a thought step to scratchpad history. Automatically parses raw text
        through ThoughtParser into structured ThoughtStepDTO and ThoughtDTO models.
        """
        if isinstance(thought, str):
            step = ThoughtParser.parse(thought)
            self.steps.append(step)
            # Create backward-compatible ThoughtDTO
            thought_dto = ThoughtDTO(
                raw_text=thought,
                thought_body=step.reasoning,
                critique="",
                confidence=step.confidence,
                parsed_decision=step.action_input.raw_output or step.action_input.code or step.action_input.command,
            )
            self.thoughts.append(thought_dto)
        elif isinstance(thought, ThoughtStepDTO):
            self.steps.append(thought)
            thought_dto = ThoughtDTO(
                raw_text=thought.reasoning,
                thought_body=thought.reasoning,
                critique="",
                confidence=thought.confidence,
                parsed_decision=thought.action_input.raw_output or thought.action_input.code or thought.action_input.command,
            )
            self.thoughts.append(thought_dto)
        elif isinstance(thought, ThoughtDTO):
            self.thoughts.append(thought)
            step = ThoughtParser.parse(thought.raw_text or thought.thought_body)
            self.steps.append(step)
        else:
            step = ThoughtParser.parse(str(thought))
            self.steps.append(step)

    @classmethod
    def get_prompt_schema_instructions(cls) -> str:
        """Returns Pydantic JSON Schema instructions tailored for small decoders."""
        return ThoughtStepDTO.get_json_schema_prompt()

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
            "steps": self.steps,
            "memory": self.retrieved_memories,
            "emotion": self.emotional_state,
            "context": self.context,
        }

    def clear(self) -> None:
        self.perception = None
        self.retrieved_memories.clear()
        self.emotional_state = None
        self.thoughts.clear()
        self.steps.clear()
        self.hypotheses.clear()
        self.intermediate_results.clear()
        self.context.clear()
        self.decision = None
        self.confidence = 0.0
