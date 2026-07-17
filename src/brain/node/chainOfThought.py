from dataclasses import dataclass, field
from typing import Any, List, Optional
from abc import ABC, abstractmethod
from src.transferDTO import ThoughtDTO

class IGoalEvaluator(ABC):
    @abstractmethod
    def evaluate(self, last_thought: ThoughtDTO, scratchpad: Any) -> bool:
        pass

class DefaultGoalEvaluator(IGoalEvaluator):
    def evaluate(self, last_thought: ThoughtDTO, scratchpad: Any) -> bool:
        if last_thought.parsed_decision is not None:
            scratchpad.set_decision(last_thought.parsed_decision, last_thought.confidence)
            return True
        return False

class MetacognitiveGoalEvaluator(IGoalEvaluator):
    def __init__(
        self,
        confidence_threshold: float = 0.85,
        decay_rate: float = 0.05,
        minimum_threshold: float = 0.55,
    ):
        self.confidence_threshold = confidence_threshold
        self.decay_rate = decay_rate
        self.minimum_threshold = minimum_threshold

    def evaluate(self, last_thought: ThoughtDTO, scratchpad: Any) -> bool:
        iteration = int(scratchpad.context.get("current_iteration", 0))
        threshold = max(
            self.minimum_threshold,
            self.confidence_threshold - (iteration * self.decay_rate),
        )

        if last_thought.parsed_decision is not None:
            if last_thought.confidence >= threshold:
                scratchpad.set_decision(last_thought.parsed_decision, last_thought.confidence)
                return True
            else:
                scratchpad.update_context(
                    "reflection_warning",
                    f"Decision proposed but confidence is too low ({last_thought.confidence:.2f} < {threshold:.2f}). Refine briefly or finalize with stronger evidence.",
                )
                
        if "wrong" in last_thought.critique.lower() or "contradict" in last_thought.critique.lower():
            scratchpad.update_context("correction_needed", True)

        if not last_thought.parse_diagnostics.parse_success:
            scratchpad.update_context(
                "format_warning",
                f"Previous response missed fields: {', '.join(last_thought.parse_diagnostics.missing_fields)}. Use the required THOUGHT/CRITIQUE/CONFIDENCE format.",
            )
            
        return False


@dataclass
class ChainOfThought:

    max_iterations: int = 5

    goal_evaluator: IGoalEvaluator = field(default_factory=MetacognitiveGoalEvaluator)

    current_iteration: int = 0

    goal_reached: bool = False

    reasoning_complete: bool = False

    thought_history: List[ThoughtDTO] = field(default_factory=list)

    def reset(self) -> None:
        self.current_iteration = 0
        self.goal_reached = False
        self.reasoning_complete = False
        self.thought_history.clear()

    def should_continue(self) -> bool:

        if self.reasoning_complete:
            return False

        if self.goal_reached:
            return False

        if self.current_iteration >= self.max_iterations:
            return False

        return True

    def update(
        self,
        thought: ThoughtDTO,
        scratchpad: Any
    ) -> None:
        self.current_iteration += 1
        self.thought_history.append(thought)
        scratchpad.update_context("current_iteration", self.current_iteration)

        if self.evaluate_goal(scratchpad):
            self.goal_reached = True
            self.reasoning_complete = True

    def evaluate_goal(
        self,
        scratchpad: Any
    ) -> bool:
        if not self.thought_history:
            return False
        return self.goal_evaluator.evaluate(self.thought_history[-1], scratchpad)

    def terminate(self) -> None:
        self.reasoning_complete = True

    def iterations_used(self) -> int:
        return self.current_iteration

    def iterations_remaining(self) -> int:
        return max(
            0,
            self.max_iterations - self.current_iteration
        )

    def progress(self) -> float:
        if self.max_iterations == 0:
            return 1.0

        return self.current_iteration / self.max_iterations

    def summary(self) -> dict:
        return {
            "iterations_used": self.current_iteration,
            "max_iterations": self.max_iterations,
            "goal_reached": self.goal_reached,
            "reasoning_complete": self.reasoning_complete,
            "thoughts_generated": len(self.thought_history),
        }
