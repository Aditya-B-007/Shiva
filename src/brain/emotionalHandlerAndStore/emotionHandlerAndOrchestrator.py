from __future__ import annotations

from typing import Any

from emotionInterface import (
    IAppraisal,
    IEmotionDynamics,
    IHomeostasis,
    IHomeostasis,
    IEmotionalMemoryTagger,
    IDreamEngine,
    IMemoryConsolidator,
    IForgettingEngine,
    IIdentityReinforcement,
    ISleepController,
    IEmotionDatabase # type: ignore
)
try:
    from src.brain.memory import MemoryEngine
except ImportError:
    try:
        from brain.memory import MemoryEngine
    except ImportError:
        from memory import MemoryEngine

class EmotionalOrchestrator:

    ##########################################################################
    # Constructor
    ##########################################################################

    def __init__(
        self,
        appraisal_engine: IAppraisal,
        emotion_engine: IEmotionDynamics,
        homeostasis_engine: IHomeostasis,
        memory_tagger: IEmotionalMemoryTagger,
        dream_engine: IDreamEngine,
        memory_consolidator: IMemoryConsolidator,
        forgetting_engine: IForgettingEngine,
        identity_engine: IIdentityReinforcement,
        sleep_controller: ISleepController,
        database: IEmotionDatabase,
        memory_engine: MemoryEngine | None = None,
    ):

        self._appraisal = appraisal_engine
        self._emotion = emotion_engine
        self._homeostasis = homeostasis_engine
        self._memory = memory_tagger
        self._dream = dream_engine
        self._consolidator = memory_consolidator
        self._forgetting = forgetting_engine
        self._identity = identity_engine
        self._sleep = sleep_controller
        self._database = database
        self._memory_engine = memory_engine

    ##########################################################################
    # Runtime
    ##########################################################################

    def perceive_event(self, event):

        appraisal = self._appraisal.evaluate(event)
        emotion = self._emotion.update(appraisal)
        homeostasis = self._homeostasis.update(appraisal,emotion)
        memory = self._memory.create_memory(event,appraisal,emotion,homeostasis)
        self._database.store_memory(memory)
        if self._memory_engine is not None:
            self._memory_engine.store(
                perception=event,
                emotion=emotion,
                homeostasis=homeostasis,
                context=self._memory_context(event, appraisal),
            )
        return emotion

    ##########################################################################
    # Queries
    ##########################################################################

    def current_emotion(self):

        return self._emotion.current_state()

    def current_homeostasis(self):

        return self._homeostasis.current_state()

    def current_identity(self):

        return self._identity.current_state()

    ##########################################################################
    # Memory
    ##########################################################################

    def retrieve_memory(self, query):

        if self._memory_engine is not None:
            return self._memory_engine.retrieve(query)
        return self._database.retrieve(query)

    ##########################################################################
    # Sleep Cycle
    ##########################################################################

    def enter_sleep(self):

        self._sleep.begin_sleep()
        if self._memory_engine is not None:
            self._memory_engine.sleep()
        replay = self._dream.generate()
        consolidated = self._consolidator.consolidate(replay)
        self._identity.reinforce(consolidated)
        self._forgetting.execute()
        self._database.optimize()
        self._sleep.finish_sleep()

    ##########################################################################
    # Wake Cycle
    ##########################################################################

    def wake(self):

        self._database.open()
        if self._memory_engine is not None:
            self._memory_engine.load()

        self._emotion.restore()

        self._homeostasis.restore()

        self._identity.restore()

    ##########################################################################
    # Persistence
    ##########################################################################

    def save(self):

        if self._memory_engine is not None:
            self._memory_engine.save()
        self._database.commit()

    def close(self):

        self._database.close()

    ##########################################################################
    # Reset
    ##########################################################################

    def reset(self):

        self._emotion.reset()

        self._homeostasis.reset()

        self._identity.reset()

    def _memory_context(self, event: Any, appraisal: Any) -> dict[str, Any]:
        context: dict[str, Any] = {}
        if hasattr(event, "event_type"):
            context["event_type"] = str(getattr(event, "event_type"))
        if hasattr(event, "source"):
            context["source"] = getattr(event, "source")
        if hasattr(event, "metadata") and isinstance(getattr(event, "metadata"), dict):
            context.update(getattr(event, "metadata"))
        for field_name in ("importance", "goal_relevance"):
            if hasattr(appraisal, field_name):
                context[field_name] = getattr(appraisal, field_name)
        return context
