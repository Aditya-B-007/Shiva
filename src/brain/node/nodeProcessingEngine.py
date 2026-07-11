class MiniProcessingEngine:

    def __init__(
        self,
        memory_engine,
        emotion_handler,
        scratchpad,
        chain_of_thought,
        reasoning_scheduler
    ):
        self._memory = memory_engine
        self._emotion = emotion_handler
        self._scratchpad = scratchpad
        self._chain = chain_of_thought

        # Shared decoder access
        self._reasoning_scheduler = reasoning_scheduler

    def process(self, perception):

        ###################################################
        # 1. Initial Cognitive State
        ###################################################

        memories = self._memory.retrieve(perception)

        emotion = self._emotion.process(
            perception,
            memories
        )

        self._scratchpad.initialize(
            perception=perception,
            memory=memories,
            emotion=emotion
        )

        ###################################################
        # 2. Thinking Loop
        ###################################################

        while self._chain.should_continue():

            ###############################################
            # Request one decoder time slice
            ###############################################

            with self._reasoning_scheduler.acquire_decoder() as decoder:

                thought = decoder.generate(
                    self._scratchpad.current_context()
                )

            ###############################################
            # Decoder released immediately here
            ###############################################

            self._scratchpad.append_thought(thought)

            self._chain.update(
                thought=thought,
                scratchpad=self._scratchpad
            )

        ###################################################
        # 3. Produce Local Decision
        ###################################################

        return self._scratchpad.final_state()
