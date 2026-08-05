from __future__ import annotations

import re
from typing import Any, Mapping

try:
    from emotionalHandlerAndStore.emotionInterface import IMemoryEngine
except ImportError:
    try:
        from ..emotionalHandlerAndStore.emotionInterface import IMemoryEngine
    except ImportError:
        from src.brain.emotionalHandlerAndStore.emotionInterface import IMemoryEngine


from .algorithms.CreditAssigner import CreditAssigner
from .algorithms.Consolidator import Consolidator
from .algorithms.DreamGenerator import DreamGenerator
from .algorithms.ForgettingModel import ForgettingModel
from .algorithms.IdentityUpdater import IdentityUpdater
from .algorithms.MemoryEncoder import MemoryEncoder
from .algorithms.SleepCycle import SleepCycle, SleepCycleResult
from src.transferDTO import MemoryGraphDTO, MemoryEdgeDTO, MemoryNodeDTO, RetrievalDTO
from .graph.MemoryEdge import AssociationType, MemoryEdge
from .graph.MemoryNode import MemoryStatus, clamp_unit
from .graph.MemoryGraph import MemoryGraph
from .repository.HybridMemoryRepository import HybridMemoryRepository


class MemoryEngine(IMemoryEngine):
    def __init__(
        self,
        graph: MemoryGraph | None = None,
        encoder: MemoryEncoder | None = None,
        sleep_cycle: SleepCycle | None = None,
        repository: MemoryRepository | None = None,
    ) -> None:
        self._repository = repository if repository is not None else HybridMemoryRepository("./memory.db", "./chroma_db")
        self._graph = graph if graph is not None else (
            self._repository.load_graph() if self._repository is not None else MemoryGraph()
        )
        self._encoder = encoder if encoder is not None else MemoryEncoder()
        self._sleep_cycle = sleep_cycle if sleep_cycle is not None else SleepCycle(
            consolidator=Consolidator(),
            forgetting_model=ForgettingModel(),
            identity_updater=IdentityUpdater(),
            dream_generator=DreamGenerator(),
        )
        self._credit_assigner = CreditAssigner()
        self._last_stored_id: str | None = None
        self._current_episode_trajectory: list[str] = []

    def store(
        self,
        perception: Any,
        emotion: Any = None,
        homeostasis: Any = None,
        context: Mapping[str, Any] | None = None,
    ) -> MemoryNodeDTO:
        node = self._encoder.encode(perception, emotion, homeostasis, context)
        self._graph.add_node(node)
        self._current_episode_trajectory.append(node.id)
        if self._last_stored_id is not None:
            self._graph.connect(
                self._last_stored_id,
                node.id,
                association_strength=0.55,
                association_type=AssociationType.TEMPORAL,
                activation_probability=0.50,
            )
        self._last_stored_id = node.id
        if self._repository is not None:
            self._repository.save_node(node)
            self._save_edges_for(node.id)
        return self._node_to_dto(node)

    def clear_trajectory(self) -> None:
        self._current_episode_trajectory.clear()

    def assign_credit_for_episode(self, reward: float) -> None:
        self._credit_assigner.assign_credit(self._graph, self._current_episode_trajectory, reward)
        self.save()
        self.clear_trajectory()

    def retrieve(self, query: Any, limit: int = 5) -> RetrievalDTO:
        if limit <= 0:
            return RetrievalDTO(query=query)
        query_text = str(query).lower()

        # 1. Vector similarity search via ChromaDB
        vector_scores: dict[str, float] = {}
        if hasattr(self._repository, "vector_search"):
            try:
                vector_scores = getattr(self._repository, "vector_search")(query_text, limit=limit * 2)
            except Exception:
                vector_scores = {}

        scored = []
        for node in self._graph.nodes:
            if node.status == MemoryStatus.PRUNED:
                continue
            text = f"{node.summary} {node.raw_content} {node.context_signature}".lower()

            # 2. Keyword lexical search score
            lexical_score = self._lexical_score(query_text, text)

            # 3. Dense vector score (from ChromaDB HNSW cosine similarity)
            vector_score = vector_scores.get(node.id, 0.0)

            # 4. Hybrid Ranking Fusion Equation
            score = (
                vector_score * 0.35
                + lexical_score * 0.25
                + node.activation * 0.15
                + node.strength * 0.10
                + node.recency * 0.10
                + node.emotional_salience * 0.05
            )
            if score > 0.0:
                scored.append((score, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        selected = [self._graph.activate(node.id, 0.05) for _, node in scored[:limit]]
        confidence = clamp_unit(scored[0][0]) if scored else 0.0
        return RetrievalDTO(
            query=query,
            memories=tuple(self._node_to_dto(node) for node in selected),
            confidence=confidence,
        )




    def _lexical_score(self, query_text: str, memory_text: str) -> float:
        query_terms = self._tokenize_and_stem(query_text)
        if not query_terms:
            return 0.0
        memory_terms = self._tokenize_and_stem(memory_text)
        if not memory_terms:
            return 0.0
        if query_text and query_text in memory_text:
            return 1.0
        return len(query_terms.intersection(memory_terms)) / len(query_terms)

    def _tokenize_and_stem(self, text: str) -> set[str]:
        terms = set()
        for word in re.findall(r"\b[a-zA-Z0-9_]{3,}\b", text.lower()):
            terms.add(self._stem(word))
        return terms

    def _stem(self, word: str) -> str:
        if word.endswith("ing") and len(word) > 5:
            return word[:-3]
        if word.endswith("ed") and len(word) > 4:
            return word[:-2]
        if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            return word[:-1]
        return word

    def sleep(self) -> SleepCycleResult:
        result = self._sleep_cycle.run(self._graph)
        self.save()
        return result

    def tick(self) -> None:
        for node in self._graph.nodes:
            if node.status != MemoryStatus.PRUNED:
                node.activation = clamp_unit(node.activation - 0.02)
                node.recency = clamp_unit(node.recency - 0.01)

    def load(self) -> None:
        if self._repository is None:
            return
        self._graph = self._repository.load_graph()
        newest = max(self._graph.nodes, key=lambda node: node.created_at, default=None)
        self._last_stored_id = newest.id if newest is not None else None

    def save(self) -> None:
        if self._repository is not None:
            self._repository.save_graph(self._graph)

    def assign_credit(self, trajectory: list[str], final_reward: float) -> None:
        self._credit_assigner.assign_credit(self._graph, trajectory, final_reward)
        self.save()

    def graph_snapshot(self) -> MemoryGraphDTO:
        return MemoryGraphDTO(
            nodes=tuple(self._node_to_dto(node) for node in self._graph.nodes),
            edges=tuple(self._edge_to_dto(edge) for edge in self._graph.edges),
        )

    def _save_edges_for(self, node_id: str) -> None:
        if self._repository is None:
            return
        for edge in self._graph.edges:
            if edge.source == node_id or edge.destination == node_id:
                self._repository.save_edge(edge)

    def _node_to_dto(self, node: MemoryNode) -> MemoryNodeDTO:
        return MemoryNodeDTO(
            id=node.id,
            raw_content=node.raw_content,
            summary=node.summary,
            modality=str(node.modality.value),
            semantic_type=str(node.semantic_type.value),
            activation=node.activation,
            strength=node.strength,
            emotional_salience=node.emotional_salience,
            identity_relevance=node.identity_relevance,
            recency=node.recency,
            status=str(node.status.value),
            access_count=node.access_count,
            created_at=node.created_at,
            last_accessed=node.last_accessed,
            context_signature=node.context_signature,
        )

    def _edge_to_dto(self, edge: MemoryEdge) -> MemoryEdgeDTO:
        return MemoryEdgeDTO(
            source=edge.source,
            destination=edge.destination,
            association_strength=edge.association_strength,
            association_type=str(edge.association_type),
            activation_probability=edge.activation_probability,
            traversal_count=edge.traversal_count,
            creation_time=edge.creation_time,
            last_traversed=edge.last_traversed,
        )
