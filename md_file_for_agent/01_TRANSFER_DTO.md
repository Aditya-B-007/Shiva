# File Breakdown: `src/transferDTO.py`

The [`src/transferDTO.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/transferDTO.py) module serves as the **Single Source of Truth (SSOT)** for all Data Transfer Objects (DTOs), event payloads, state vectors, memory graph schema, and Pydantic communication schemas across the entire Shiva cognitive architecture.

---

## Line-by-Line & Section-by-Section Analysis

### Section 1: Core Token & Latent Representation DTOs (Lines 16–51)

- **Lines 1–10**: Imports PyTorch (`torch`), NumPy (`numpy`), `dataclass`, `field`, `datetime`, `Enum`, typing generics, and UUID utilities.
- **Lines 16–21 (`Tokens`)**:
  - `values: torch.Tensor`: Shape `(batch_size, num_features, vector_size)`. Holds raw encoded tensor groups.
  - `names: List[str]`: Identifies feature token strings matching tensor indices.
- **Lines 23–43 (`TokenBundle`)**:
  - Encapsulates dictionary of `Tokens` indexed by group name with automatic creation timestamp.
  - `@property def tensor(self)` (Lines 29–34): Concatenates active token tensors across groups along dimension 1. Raises `ValueError` if no active tokens are present.
  - `@property def names(self)` (Lines 36–41): Flattens feature names across token groups into a single list.
- **Lines 44–50 (`Latent`)**:
  - `vector: torch.Tensor`: Pooled latent vector representation of state.
  - `features: Dict[str, torch.Tensor]`: Map of individual feature names to post-attention tensors.

---

### Section 2: Perception & Observation DTOs (Lines 56–101)

- **Lines 56–60 (`ObservationKind`)**: String Enum specifying observation categories (`TEXT`, `STRUCTURED`, `ERROR`, `UNKNOWN`).
- **Lines 63–72 (`PerceptionObservationDTO`)**:
  - Encapsulates output captured from workspace sensing hooks or system devices.
  - Contains `device` identifier, `kind`, textual `summary`, raw `payload`, `payload_size`, `metadata` dictionary, and `captured_at` timestamp.
- **Lines 75–94 (`PerceptionBundleDTO`)**:
  - Aggregates the original user prompt `query` with a list of captured `PerceptionObservationDTO` instances.
  - `has_observations` property (Lines 80–82): Returns boolean flag.
  - `observation_names()` (Lines 84–85): Returns list of device identifiers attached to observations.
  - `errors()` (Lines 87–93): Filters observations to return only error occurrences.
- **Lines 95–101 (`PerceptionCaptureRequestDTO`)**:
  - Frozen dataclass representing a request sent to capture data from a specified device with custom arguments.

---

### Section 3: Memory Node & Graph DTOs (Lines 106–150)

- **Lines 106–124 (`MemoryNodeDTO`)**:
  - Immutable memory node model representing a single episodic, semantic, or working memory record.
  - Attributes: `id`, `raw_content`, `summary`, `modality`, `semantic_type`, `activation` float, `strength` float, `emotional_salience` float, `identity_relevance` float, `recency` float, `status`, `access_count`, `created_at`, `last_accessed`, and `context_signature`.
- **Lines 125–137 (`MemoryEdgeDTO`)**:
  - Immutable relational link connecting `source` and `destination` node IDs.
  - Tracks `association_strength`, `association_type` (e.g. `causal`, `temporal`, `semantic`), `activation_probability`, `traversal_count`, `creation_time`, and `last_traversed`.
- **Lines 138–143 (`MemoryGraphDTO`)**: Immutable container storing tuples of `MemoryNodeDTO` and `MemoryEdgeDTO`.
- **Lines 144–150 (`RetrievalDTO`)**: Encapsulates hybrid RAG search result tuple of memory nodes with scalar `confidence`.

---

### Section 4: Cognitive Brain & Reasoning DTOs (Lines 157–294)

- **Lines 157–164 (`ActionInputDTO`)**:
  - Pydantic `BaseModel` specifying structured parameters for actions parsed from LLM outputs.
  - `code`: Executable Python script string.
  - `command`: Terminal CLI command string.
  - `params`: Parameter dictionary.
  - `raw_output`: Buffer for raw unparsed text or decisions.
- **Lines 166–181 (`ThoughtStepDTO`)**:
  - Primary structured cognitive output model emitted by the `ThoughtParser`.
  - Fields: `reasoning` (step-by-step reflection/critique), `action` (`scratchpad_note`, `execute_code`, `decision`), `action_input` (`ActionInputDTO`), `confidence` (clamped float `0.0`–`1.0`).
  - `@classmethod get_json_schema_prompt()` (Lines 174–180): Dynamically dumps `model_json_schema()` to format strict JSON prompt instructions for SLMs.
- **Lines 184–190 (`BrainErrorDTO`)**: Captures errors inside cognitive components with `recoverable` flag.
- **Lines 193–197 (`ThoughtParseDiagnosticsDTO`)**: Diagnostics tracking parser success, missing fields, and warnings.
- **Lines 200–209 (`ThoughtDTO`)**: Backward-compatible thought object containing `raw_text`, `thought_body`, `critique`, `confidence`, `parsed_decision`, and diagnostic metadata.
- **Lines 211–221 (`ReasoningContextDTO`)**: Assembles current working memory, active perception bundle, retrieved memories, emotional state, thought history, and hypotheses for decoder prompt generation.
- **Lines 222–233 (`NodeReasoningResultDTO`)**: Output produced by a single node reasoning cycle (`decision`, `confidence`, `thought_history`, `iterations_used`, `max_iterations`, `goal_reached`, `errors`, `timed_out`).
- **Lines 235–242 (`ColumnResultDTO`)**: Result wrapper from a specialized Cortical Column (`column_id`, `role`, `result`, `error`, `duration_ms`).
- **Lines 244–256 (`MothershipResponseDTO`)**: Final response object produced by executive Mothership arbitration.
- **Lines 258–266 (`BrainHealthDTO`)**: System health status tracking online states of cognitive core, decoder, memory engine, and homeostasis.
- **Lines 268–294 (`EncoderInputDTO`, `EncoderOutputDTO`, `DecoderInputDTO`, `DecoderOutputDTO`)**: Tensor I/O wrappers bridging PyTorch tensors and NumPy arrays between neural transformers and memory subsystems.

---

### Section 5: Emotional Dynamics, Homeostasis & Appraisal DTOs (Lines 299–481)

- **Lines 299–325 (`EventType`)**: Comprehensive Enum categorizing cognitive, sensory, memory, and lifecycle events (`PERCEPTION`, `MEMORY_RETRIEVAL`, `GOAL_COMPLETED`, `ACTION_FAILED`, `SLEEP`, `WAKE`, etc.).
- **Lines 327–334 (`Event[T]`)**: Generic dataclass wrapping typed payloads with UUIDs and source identifiers.
- **Lines 337–344 (`PerceptionDTO`)**: Legacy perception model carrying text embeddings and sensor data.
- **Lines 347–358 (`EnvironmentDTO`)**: Device state telemetry (battery, CPU, GPU, memory, time, location).
- **Lines 361–370 (`GoalDTO`)**: Goal tracking structure with priority, progress (`0.0`–`1.0`), and deadline.
- **Lines 373–390 (`EmotionDTO`)**: Multi-dimensional affective state vector (`joy`, `sadness`, `fear`, `anger`, `surprise`, `disgust`, `trust`, `anticipation`, `curiosity`, `confidence`, `frustration`, `motivation`, `uncertainty`, `dominant_emotion`).
- **Lines 393–404 (`HomeostasisDTO`)**: Homeostatic state values (`fatigue`, `stress`, `cognitive_load`, `focus`, `curiosity_drive`, `novelty_hunger`, `reward_satisfaction`, `social_need`, `stability_score`).
- **Lines 407–418 (`IdentityDTO`)**: System identity parameters (`name`, `beliefs`, `values`, `preferences`, `skills`, `personality_traits`).
- **Lines 421–428 (`MemoryDTO`)**: Aggregated container for episodic, semantic, working, and retrieved memories.
- **Lines 431–445 (`AppraisalDTO`)**: Cognitive appraisal vector (`novelty`, `threat`, `reward`, `controllability`, `urgency`, `familiarity`, `confidence`, `prediction_error`, `importance`, `goal_relevance`, `agency`, `information_gain`).
- **Lines 449–481 (`FeatureBundle`, `NumericalFeatureVector`, `FeatureTokenSequence`, `CognitiveLatent`)**: Composite feature bundles and latent representation wrappers used by FT-Transformer appraisal networks.
