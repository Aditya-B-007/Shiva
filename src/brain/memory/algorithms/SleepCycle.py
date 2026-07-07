from __future__ import annotations

from dataclasses import dataclass

from ..graph.MemoryGraph import MemoryGraph
from .Consolidator import Consolidator
from .DreamGenerator import DreamGenerator
from .ForgettingModel import ForgettingModel
from .IdentityUpdater import IdentityUpdater


@dataclass(frozen=True, slots=True)
class SleepCycleResult:
    consolidated_ids: tuple[str, ...]
    forgotten_ids: tuple[str, ...]
    identity_updated_ids: tuple[str, ...]
    dream_sequence_ids: tuple[str, ...]


class SleepCycle:
    def __init__(
        self,
        consolidator: Consolidator,
        forgetting_model: ForgettingModel,
        identity_updater: IdentityUpdater,
        dream_generator: DreamGenerator,
    ) -> None:
        self._consolidator = consolidator
        self._forgetting_model = forgetting_model
        self._identity_updater = identity_updater
        self._dream_generator = dream_generator

    def run(self, graph: MemoryGraph) -> SleepCycleResult:
        consolidated = self._consolidator.consolidate(graph)
        forgotten = self._forgetting_model.apply(graph)
        identity_updated = self._identity_updater.update(graph)
        dream_sequence = self._dream_generator.generate(graph)
        return SleepCycleResult(
            consolidated_ids=consolidated,
            forgotten_ids=forgotten,
            identity_updated_ids=identity_updated,
            dream_sequence_ids=tuple(node.id for node in dream_sequence),
        )
