# File Breakdown: Cognitive Transformer & Node Reasoning Subsystems

This document provides a line-by-line breakdown of neural transformer models, resilient thought parsing, working memory scratchpads, chain-of-thought goal evaluation, and single-node cognitive orchestration engines.

---

## 1. Causal Language Model Decoder ([`src/brain/transformer/Decoder.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/transformer/Decoder.py))

- **Lines 18–74 (`Decoder.__init__`)**:
  - Model identifier: `HuggingFaceTB/SmolLM2-360M-Instruct`. Local disk cache path: `models/shiva-decoder`.
  - Automatically selects device (`cuda`, `mps`, or `cpu`).
  - Lazy-loads or downloads `AutoTokenizer` and `AutoModelForCausalLM`. Sets pad token to `eos_token`. Initializes `projection_layer` to `None`.
- **Lines 75–94 (`input`)**: Projects arbitrary continuous input vectors or NumPy arrays (`vector_input`) to match the decoder hidden size (`960`) via a dynamic linear layer (`nn.Linear(current_input_dim, target_dim)`).
- **Lines 96–120 (`process` & `output`)**: Executes forward pass under `torch.no_grad()` and returns `DecoderOutputDTO` containing PyTorch tensors and NumPy arrays.
- **Lines 121–176 (`execute_sandbox_script`)**:
  - Compiles and executes Python 3 code strings inside an isolated runtime scope.
  - Exposes standard Python modules (`math`, `json`, `re`, `pathlib`, `shutil`, `glob`, `datetime`, `subprocess`, `urllib`, etc.) and sets `WORKSPACE_DIR`.
  - Captures `stdout` and `stderr` using `contextlib.redirect_stdout`. Returns execution logs or sandbox error stack traces.
- **Lines 177–314 (`generateDecision`)**:
  - Formats system instructions and user context (`perception`, `emotion`, `memories`, `hypotheses`, `thoughts`, `context`) into model chat format via `tokenizer.apply_chat_template`.
  - Injecting Latent Actions (Lines 245–280): If `latent_action` vector is provided by SAC RL policies, projects vector to hidden dimension and prepends `latent_embed` to `inputs_embeds` ahead of token embeddings.
  - Executes text generation via `model.generate()`. Decodes raw output text and parses it through `parse_thought_text(raw_text)`.
  - Automated Sandbox Execution (Lines 293–310): If decision body contains a ```python block, automatically executes the script in the sandbox and appends `--- SANDBOX EXECUTION OUTPUT ---` directly to the decision string.

---

## 2. Feature Transformer Encoder ([`src/brain/transformer/Encoder.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/transformer/Encoder.py))

- **Lines 8–24 (`Encoder.__init__`)**: Uses `google-bert/bert-large-uncased` cached under `models/shiva-encoder`. Detects device accelerator (`cuda`/`mps`/`cpu`).
- **Lines 25–50 (`_ensure_model_loaded`)**: Lazy model loader. Handles offline loading (`local_files_only=True`) with fallback download (`local_files_only=False`).
- **Lines 51–60 (`input`)**: Tokenizes input text strings with padding, truncation (`max_length=512`), returning PyTorch tensor dict.
- **Lines 62–70 (`process`)**: Executes forward pass under `torch.no_grad()`.
- **Lines 71–86 (`Output`)**: Extracts `last_hidden_state` and `pooler_output`, returning PyTorch and NumPy arrays in `EncoderOutputDTO`.

---

## 3. Multi-Tier Resilient Thought Parser ([`src/brain/transformer/thought_parser.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/transformer/thought_parser.py))

- **Lines 10–23 (`ThoughtParser`)**: Multi-tier parsing engine designed to guarantee zero unhandled exceptions on malformed LLM streams.
- **Lines 21–64 (`parse(raw_text)`)**: Public entry point orchestrating 5 parsing tiers:
  1. **Stage 1 (Lines 70–85 - `_extract_xml_think_blocks`)**: Extracts text inside `<think>...</think>` tags (or unclosed `<think>` tags emitted by reasoning models).
  2. **Stage 2 (Lines 87–95 - `_extract_markdown_json`)**: Extracts JSON candidates inside ```json ... ``` or ``` ... ``` code blocks.
  3. **Stage 3 (Lines 97–101 - `_extract_regex_json`)**: Regex search matching raw JSON objects (`\{[\s\S]*\}`).
  4. **Stage 4 (Lines 103–151 - `_pre_clean_json` & `_try_parse_pydantic`)**: Fixes trailing commas, replaces Python literal tokens (`True`/`False`/`None` $\rightarrow$ `true`/`false`/`null`), and validates with Pydantic `ThoughtStepDTO`.
  5. **Stage 5 (Lines 153–192 - `_parse_legacy_tags`)**: Parses legacy `THOUGHT:`, `CRITIQUE:`, `CONFIDENCE:`, `DECISION:` text tags.
  - **Emergency Fallback (Lines 56–63)**: If all tiers fail, returns `ThoughtStepDTO(reasoning=..., action="scratchpad_note", confidence=0.3)`.
- **Lines 204–230 (`parse_thought_text`)**: Wrapper converting `ThoughtStepDTO` to backward-compatible `ThoughtDTO` with diagnostic diagnostics.

---

## 4. Working Memory & ScratchPad ([`src/brain/node/scratchPad.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/node/scratchPad.py))

- **Lines 7–19 (`ScratchPad`)**: Dataclass holding working memory during a reasoning episode (`perception`, `retrieved_memories`, `emotional_state`, `thoughts`, `steps`, `intermediate_results`, `hypotheses`, `context`, `decision`, `confidence`).
- **Lines 30–64 (`append_thought`)**: Accepts string, `ThoughtStepDTO`, or `ThoughtDTO` objects. Automatically passes raw text strings through `ThoughtParser.parse()`.
- **Lines 65–68 (`get_prompt_schema_instructions`)**: Retrieves Pydantic JSON Schema prompt instructions for injection into SLM prompts.
- **Lines 83–91 (`current_context`)**: Bundles state into a `ReasoningContextDTO` for decoder input.
- **Lines 112–123 (`clear`)**: Resets all working lists and attributes.

---

## 5. Chain-of-Thought & Goal Evaluation ([`src/brain/node/chainOfThought.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/node/chainOfThought.py))

- **Lines 6–16 (`IGoalEvaluator` & `DefaultGoalEvaluator`)**: Abstract goal evaluation interface and default evaluator that completes reasoning whenever `parsed_decision` is present.
- **Lines 18–56 (`MetacognitiveGoalEvaluator`)**:
  - Dynamically decays confidence threshold per iteration ($\text{threshold} = \max(\text{min}, \text{initial} - \text{iteration} \times \text{decay})$).
  - Flags reflection warnings if decision confidence is below threshold, flags correction flags on critiques, and flags formatting warnings on diagnostic failures.
- **Lines 59–139 (`ChainOfThought`)**:
  - Tracks iteration count (`max_iterations=5`), goal completion status, and thought history.
  - `should_continue()` (Lines 79–90): Returns `False` if goal is reached, reasoning is complete, or max iterations are exceeded.

---

## 6. Node Processing Engine ([`src/brain/node/nodeProcessingEngine.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/node/nodeProcessingEngine.py))

- **Lines 6–18 (`ReasoningScheduler`)**: Thread-safe lock manager (`threading.Lock`) guarding decoder time slices across multi-threaded operations.
- **Lines 20–107 (`nodeProcessingEngine`)**:
  - Single-node cognitive loop coordinator.
  - `process()` method:
    1. Retrieves hybrid RAG memories from `MemoryEngine`.
    2. Perceives homeostatic/emotional state from `emotion_handler`.
    3. Initializes `ScratchPad` and appends seed thoughts.
    4. Executes loop while `chain.should_continue()`: acquires decoder lock slice, calls `generateDecision()`, appends thought step to scratchpad, and updates `ChainOfThought`.
    5. Returns `NodeReasoningResultDTO`.
