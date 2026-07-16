import unittest

from src.brain.BrainDTO import ThoughtDTO
from src.brain.emotionalHandlerAndStore.Homeostasis import (
    EulerIntegrator,
    GraphDerivative,
    HomeostasisState,
    HomeostasisVariable,
)
from src.brain.memory.MemoryEngine import MemoryEngine
from src.brain.node.chainOfThought import ChainOfThought
from src.brain.node.nodeProcessingEngine import ReasoningScheduler, nodeProcessingEngine
from src.brain.node.scratchPad import ScratchPad
from src.brain.transformer.thought_parser import parse_thought_text


class DummyDecoder:
    def __init__(self):
        self.kwargs = None

    def generateDecision(self, context, **kwargs):
        self.kwargs = kwargs
        assert "current_iteration" in context.context or not context.thoughts
        return ThoughtDTO(
            raw_text="THOUGHT: done\nCRITIQUE: none\nCONFIDENCE: 0.91\nDECISION: proceed",
            thought_body="done",
            critique="none",
            confidence=0.91,
            parsed_decision="proceed",
        )


class DummyMemory:
    def retrieve(self, perception):
        return []


class DummyEmotion:
    def current_emotion(self):
        return {"state": "neutral"}


class BrainV1HardeningTests(unittest.TestCase):
    def test_thought_parser_accepts_markdown_lowercase_and_reordered_tags(self):
        thought = parse_thought_text(
            "**confidence:** 87\n"
            "**decision:** continue\n"
            "**thought:** inspect the memory first\n"
            "**critique:** may be missing sensor context"
        )

        self.assertEqual(thought.thought_body, "inspect the memory first")
        self.assertEqual(thought.critique, "may be missing sensor context")
        self.assertEqual(thought.parsed_decision, "continue")
        self.assertAlmostEqual(thought.confidence, 0.87)
        self.assertTrue(thought.parse_diagnostics.warnings)

    def test_node_engine_passes_decoder_policy(self):
        decoder = DummyDecoder()
        engine = nodeProcessingEngine(
            memory_engine=DummyMemory(),
            emotion_handler=DummyEmotion(),
            scratchpad=ScratchPad(),
            chain_of_thought=ChainOfThought(max_iterations=2),
            reasoning_scheduler=ReasoningScheduler(decoder),
        )

        result = engine.process("input", decoder_kwargs={"temperature": 0.2, "max_new_tokens": 32})

        self.assertEqual(result.decision, "proceed")
        self.assertEqual(decoder.kwargs["temperature"], 0.2)
        self.assertEqual(decoder.kwargs["max_new_tokens"], 32)
        self.assertFalse(result.errors)

    def test_memory_retrieval_uses_token_overlap(self):
        memory = MemoryEngine()
        memory.store("camera captured moving objects")

        result = memory.retrieve("capturing object", limit=1)

        self.assertEqual(len(result.memories), 1)
        self.assertGreater(result.confidence, 0.0)

    def test_homeostasis_soft_boundary_does_not_hard_clip_positive_derivative(self):
        state = HomeostasisState(
            values={
                HomeostasisVariable.ENERGY.value: 0.99,
                HomeostasisVariable.METASTABILITY.value: 0.3,
            }
        )
        derivative = GraphDerivative(
            values={
                HomeostasisVariable.ENERGY.value: 10.0,
                HomeostasisVariable.METASTABILITY.value: 0.4,
            },
            interaction={},
            restoration={},
            external={},
        )

        next_state = EulerIntegrator().integrate(state, derivative, 0.05)

        self.assertLess(next_state.value(HomeostasisVariable.ENERGY), 1.0)
        self.assertGreater(next_state.value(HomeostasisVariable.ENERGY), 0.99)


if __name__ == "__main__":
    unittest.main()
