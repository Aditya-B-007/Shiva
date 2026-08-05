# File Breakdown: Multi-Tier Memory Engine & Persistence

This document provides a comprehensive line-by-line analysis of Shiva's hybrid vector-graph memory architecture, storage persistence layers, and memory dynamics algorithms.

---

## 1. Primary Memory Orchestrator ([`src/brain/memory/MemoryEngine.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/MemoryEngine.py))

- **Class `MemoryEngine`**: Central coordinator for storing, retrieving, consolidating, and reinforcing memory nodes across ChromaDB vector collections and SQLite relational graphs.
- **`store()`**: Accepts perception bundles, emotional states, and homeostatic vectors; constructs `MemoryNode` instances, calculates initial emotional salience, dual-writes nodes via `HybridMemoryRepository`, and creates graph edges.
- **`retrieve(query, limit=5)`**: Implements the Hybrid RAG Search Scoring algorithm:
  $$\text{Score}(N) = 0.35 \cdot S_{\text{vector}} + 0.25 \cdot S_{\text{keyword}} + 0.15 \cdot \text{Activation} + 0.10 \cdot \text{Strength} + 0.10 \cdot \text{Recency} + 0.05 \cdot \text{Salience}$$
  - $S_{\text{vector}}$: Cosine similarity score retrieved from ChromaDB HNSW vector index (`shiva_nodes`).
  - $S_{\text{keyword}}$: Stemmed keyword intersection ratio.
  - Spreading Activation: Traverses topological graph edges (`MemoryGraph`) to activate neighbor nodes.
- **`sleep()`**: Invokes `SleepCycle.run()`, consolidating short-term scratchpad entries into long-term graph nodes and running synthetic dream trajectory generation.
- **`assign_credit_for_episode(reward)`**: Calls `CreditAssigner` to propagate Monte Carlo temporal reinforcement:
  $$\Delta V(S) = \alpha (G - V(S))$$

---

## 2. Topological Memory Graph Core ([`src/brain/memory/graph/`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/graph/))

### `MemoryNode.py`
- Represents a single memory entity. Attributes include `id`, `raw_content`, `summary`, `modality` (`episodic`, `semantic`, `procedural`), `activation` state ($0.0$–$1.0$), `strength` (resistance to decay), `emotional_salience`, `identity_relevance`, `recency`, and access metrics.

### `MemoryEdge.py`
- Represents an associative topological edge connecting two `MemoryNode` instances. Attributes: `source`, `destination`, `association_strength`, `association_type` (`causal`, `temporal`, `semantic`, `hierarchical`), `activation_probability`, and `traversal_count`.

### `MemoryGraph.py`
- Graph traversal engine. Performs Breadth-First Search (BFS) spreading activation across neighbor edges. Decays dormant activation states over time and prunes weak association links.

---

## 3. Persistence Repository Architecture ([`src/brain/memory/repository/`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/repository/))

### Abstract Repository Interface ([`MemoryRepository.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/repository/MemoryRepository.py))
- Defines abstract methods `save_node()`, `get_node()`, `query_similar()`, `save_edge()`, `get_edges()`, and `delete_node()`.

### Vector Store Persistence ([`ChromaMemoryRepository.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/repository/ChromaMemoryRepository.py))
- Wraps ChromaDB vector client persisting to disk directory `chroma_db/`.
- Stores composite document text embeddings (`summary` + `context_signature` + `raw_content`) in the `shiva_nodes` HNSW collection. Converts cosine distance to normalized similarity scores.

### Relational Graph Persistence ([`SQLiteMemoryRepository.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/repository/SQLiteMemoryRepository.py))
- Connects to `memory.db` via `SQLiteManager`. Stores relational edge tables, node activation levels, traversal counts, and credit log records.

### Dual-Write Coordinator ([`HybridMemoryRepository.py`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/repository/HybridMemoryRepository.py))
- Coordinates concurrent dual-writes: sends vector payloads to ChromaDB and topological graph structures to SQLite. Performs unified hybrid query execution.

---

## 4. Memory Dynamics & Reinforcement Algorithms ([`src/brain/memory/algorithms/`](file:///Users/veenadhruva/Desktop/Aditya_projects/Working/Shiva/src/brain/memory/algorithms/))

### `Consolidator.py`
- Short-to-long-term consolidation engine. Scans working memory scratchpads, clusters related episodic thoughts, synthesizes semantic node summaries, and promotes nodes to long-term storage.

### `CreditAssigner.py`
- Reinforcement learning credit assignment. Traces trajectory nodes activated during a reasoning episode and updates node strength values using Monte Carlo temporal difference updates based on final task reward ($G$).

### `DreamGenerator.py`
- Synthetic trajectory generator. During background sleep cycles, samples random memory clusters and constructs synthetic reasoning episodes to stabilize vector space embeddings and strengthen association links.

### `ForgettingModel.py`
- Ebbinghaus decay model. Calculates memory retention decay:
  $$R = e^{-\frac{t}{S}}$$
  Prunes dormant nodes with activation and strength falling below minimal thresholds.

### `IdentityUpdater.py`
- High-level identity refinement model. Updates agent personality traits, core values, and belief systems based on consolidated long-term episodic outcomes.

### `SleepCycle.py`
- Background orchestrator. Sequentially executes forgetting decay, memory consolidation, dream generation, and index optimization during idle user periods.
