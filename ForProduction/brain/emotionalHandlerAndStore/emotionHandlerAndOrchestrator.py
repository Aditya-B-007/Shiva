from __future__ import annotations

from emotionInterfaces import (
    IAppraisal,
    IEmotionDynamics,
    IHomeostasis,
    IEmotionalMemoryTagger,
    IDreamEngine,
    IMemoryConsolidator,
    IForgettingEngine,
    IIdentityReinforcement,
    ISleepController,
    IEmotionDatabase
)
"""
emotion/
│
├── EmotionalOrchestrator.py
│
├── EmotionInterfaces.py
│
├── AppraisalEngine.py
├── EmotionDynamicsEngine.py
├── HomeostasisEngine.py
├── EmotionalMemoryTagger.py
├── DreamEngine.py
├── MemoryConsolidator.py
├── ForgettingEngine.py
├── IdentityReinforcement.py
├── SleepController.py
│
├── EmotionStorage.py
│
└── __init__.py
"""

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

    ##########################################################################
    # Runtime
    ##########################################################################

    def perceive_event(self, event):

        appraisal = self._appraisal.evaluate(event)
        emotion = self._emotion.update(appraisal)
        homeostasis = self._homeostasis.update(appraisal,emotion)
        memory = self._memory.create_memory(event,appraisal,emotion,homeostasis)
        self._database.store_memory(memory)
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

        return self._database.retrieve(query)

    ##########################################################################
    # Sleep Cycle
    ##########################################################################

    def enter_sleep(self):

        self._sleep.begin_sleep()
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

        self._emotion.restore()

        self._homeostasis.restore()

        self._identity.restore()

    ##########################################################################
    # Persistence
    ##########################################################################

    def save(self):

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
