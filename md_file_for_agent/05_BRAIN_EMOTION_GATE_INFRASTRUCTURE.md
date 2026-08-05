# File Breakdown: Regulation, Transactional Gates & Infrastructure

This document details the dynamic emotional appraisal networks, homeostatic regulation systems, transactional memory write gates, thread-safe SQLite connection managers, and asynchronous worker queue infrastructure.

---

## 1. Dynamic Emotional & Homeostatic Regulation ([`src/brain/emotionalHandlerAndStore/`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/emotionalHandlerAndStore/))

### `AppraisalEngine.py`
- Implements cognitive appraisal theory using PyTorch neural networks:
  - `FeatureExtractor`: Extracts numerical and categorical features from incoming `Event` bundles.
  - `FTTransformerFeatureEmbedding`: Projects heterogeneous event features into unified continuous embedding vectors (`vector_size=192`).
  - `CognitiveStateEncoder`: Multi-head self-attention encoder (`TransformerConfig`) computing contextual feature interactions.
  - `AppraisalNetwork`: Multi-layer perceptron mapping contextual state embeddings to a 13-dimensional `AppraisalDTO` vector (`novelty`, `threat`, `reward`, `controllability`, `urgency`, `goal_relevance`, etc.).

### `EmotionDynamicsEngine.py`
- Differential dynamics model updating the 13-dimensional `EmotionDTO` vector (`joy`, `sadness`, `fear`, `anger`, `curiosity`, `frustration`, `confidence`, etc.) based on cognitive appraisal feedback and decay rates.

### `Homeostasis.py`
- Systemic equilibrium model tracking internal homeostatic metrics:
  - Metrics: `fatigue` ($0.0$–$1.0$), `stress` ($0.0$–$1.0$), `cognitive_load` ($0.0$–$1.0$), `focus` ($0.0$–$1.0$), `curiosity_drive`, and `stability_score`.
  - Updates homeostatic metrics dynamically based on task duration, processing complexity, error frequencies, and rest intervals.

### `emotionHandlerAndOrchestrator.py` (`EmotionalOrchestrator`)
- Master coordinator connecting `AppraisalEngine`, `EmotionDynamicsEngine`, `Homeostasis`, and `MemoryEngine`.
- Receives events (`perceive_event()`), computes appraisals, updates emotional and homeostatic states, and persists state snapshots to memory.

---

## 2. Transactional Gate Architecture & Promotion ([`src/brain/gate/`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/gate/))

### Gate Core Interface ([`Gate.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/gate/Gate.py))
- Base class enforcing transactional memory write semantics. Manages write request staging, validation, commit, and rollback capabilities.

### Working Memory Gate ([`ScratchPadGate.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/gate/ScratchPadGate.py))
- Transactional buffer for transient thought steps and intermediate outputs created during reasoning episodes.

### Long-Term Memory Gate ([`MemoryGate.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/gate/MemoryGate.py))
- Transactional promotion gate governing commits to long-term database storage.

### Services & Promotion Policies ([`src/brain/gate/services/`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/gate/services/))
- `PromotionPolicy.py`: Evaluates whether a scratchpad entry satisfies quality thresholds (confidence score, access count, emotional salience, non-redundancy) for promotion to long-term memory.
- `TransactionService.py`: Orchestrates multi-step transaction commit pipelines across repositories.

---

## 3. Infrastructure & Persistence Mechanics ([`src/brain/infrastructure/`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/infrastructure/))

### Thread-Safe SQLite Connection Manager ([`SQLiteManager.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/infrastructure/sqlite/SQLiteManager.py))
- Manages thread-local SQLite database connections using PyTorch/Python threading locals (`threading.local()`).
- Enables Write-Ahead Logging (`PRAGMA journal_mode=WAL;`), foreign key constraints (`PRAGMA foreign_keys=ON;`), and busy timeouts (`PRAGMA busy_timeout=5000;`) to prevent database lock contention during concurrent swarm operations.

### Scratch & Memory Repositories ([`SQLiteScratchRepository.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/infrastructure/sqlite/SQLiteScratchRepository.py) & [`SQLiteMemoryRepository.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/infrastructure/sqlite/SQLiteMemoryRepository.py))
- Implements raw SQL queries for table creation, transactional inserts, updates, and schema migrations.

### Asynchronous Queue & Background Worker ([`src/brain/infrastructure/queue/`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/infrastructure/queue/))
- `QueueManager.py`: Thread-safe, bounded task queue manager using `queue.Queue`.
- `GateWorker.py`: Background worker thread consuming queued transactional write requests asynchronously to prevent blocking real-time inference loops.
