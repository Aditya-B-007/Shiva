from __future__ import annotations

from typing import Any

try:
    from emotionInterface import (
        IAppraisal,
        IEmotionDynamics,
        IHomeostasis,
        IMemoryEngine,
    )
    from emotionalContract import EmotionDTO, HomeostasisDTO
except ImportError:
    try:
        from .emotionInterface import (
            IAppraisal,
            IEmotionDynamics,
            IHomeostasis,
            IMemoryEngine,
        )
        from .emotionalContract import EmotionDTO, HomeostasisDTO
    except ImportError:
        from src.brain.emotionalHandlerAndStore.emotionInterface import (
            IAppraisal,
            IEmotionDynamics,
            IHomeostasis,
            IMemoryEngine,
        )
        from src.brain.emotionalHandlerAndStore.emotionalContract import (
            EmotionDTO,
            HomeostasisDTO,
        )


class EmotionalOrchestrator:

    ##########################################################################
    # Constructor
    ##########################################################################

    def __init__(
        self,
        appraisal_engine: IAppraisal,
        emotion_engine: IEmotionDynamics,
        homeostasis_engine: IHomeostasis,
        memory_engine: IMemoryEngine,
    ):
        self._appraisal = appraisal_engine
        self._emotion = emotion_engine
        self._homeostasis = homeostasis_engine
        self._memory_engine = memory_engine
        self._current_emotion = EmotionDTO()

    ##########################################################################
    # Runtime
    ##########################################################################

    def perceive_event(self, event: Any) -> EmotionDTO:
        # Wrap string/non-FeatureBundle perceptions into a FeatureBundle to satisfy AppraisalEngine requirements
        from src.brain.emotionalHandlerAndStore.emotionalContract import FeatureBundle, Event, EventType, PerceptionDTO
        if not isinstance(event, FeatureBundle):
            evt = Event(event_type=EventType.PERCEPTION, payload=str(event), source="perception")
            perc = PerceptionDTO(text=str(event))
            event = FeatureBundle(event=evt, perception=perc)

        appraisal = self._appraisal.evaluate(event)
        homeostasis = self._homeostasis.current_state()
        
        # update emotion
        emotion = self._emotion.evaluate(appraisal, self._current_emotion, homeostasis)
        self._current_emotion = emotion
        
        # update homeostasis
        self._homeostasis.update(appraisal, emotion)
        
        # store memory
        self._memory_engine.store(
            perception=event,
            emotion=emotion,
            homeostasis=self._homeostasis.current_state(),
            context=self._memory_context(event, appraisal),
        )
        return emotion

    ##########################################################################
    # Queries
    ##########################################################################

    def current_emotion(self) -> EmotionDTO:
        return self._current_emotion

    def current_homeostasis(self) -> HomeostasisDTO:
        return self._homeostasis.current_state()

    def current_identity(self) -> Any:
        return None

    ##########################################################################
    # Memory
    ##########################################################################

    def retrieve_memory(self, query: Any, limit: int = 5) -> Any:
        return self._memory_engine.retrieve(query, limit)

    ##########################################################################
    # Sleep Cycle
    ##########################################################################

    def enter_sleep(self) -> Any:
        return self._memory_engine.sleep()

    ##########################################################################
    # Wake Cycle
    ##########################################################################

    def wake(self) -> None:
        self._memory_engine.load()
        self._homeostasis.restore()
        self._current_emotion = EmotionDTO()

    ##########################################################################
    # Persistence
    ##########################################################################

    def save(self) -> None:
        self._memory_engine.save()

    def close(self) -> None:
        pass

    ##########################################################################
    # Reset
    ##########################################################################

    def reset(self) -> None:
        self._current_emotion = EmotionDTO()
        self._homeostasis.reset()

    ##########################################################################
    # Helpers
    ##########################################################################

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
