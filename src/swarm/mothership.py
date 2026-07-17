import math
import logging
import time
import threading
from typing import Any, List, Dict, Iterable, Optional
from src.swarm.cells import AnalyticalColumn, CreativeColumn, RiskColumn, VerificationColumn, CorticalColumn
from src.brain.node.nodeDTOs import BrainErrorDTO, NodeReasoningResultDTO, ThoughtDTO
from src.input.hook.perception import (
    PerceptionBundleDTO,
    PerceptionCaptureRequestDTO,
    PerceptionObservationDTO,
    PerceptionObservationFactory,
    PerceptionPromptFormatter,
)
from src.swarm.swarmDTOs import MothershipResponseDTO

logger = logging.getLogger("shiva.swarm.mothership")

class CognitiveStabilityRegulator:
    def __init__(self):
        self.theta = 0.0          # Angular deviation representing cognitive instability (rad)
        self.theta_dot = 0.0      # Rate of instability change (rad/s)
        self.gravity = 9.81
        self.length = 1.0
        self.damping = 0.15

    def apply_cognitive_perturbations(self, uncertainty: float, stress: float, conflict: float, dt: float = 0.05):
        # Cognitive disturbances act as destabilizing forces pushing the pendulum away from theta=0
        disturbance_force = (uncertainty * 1.5) + (stress * 1.0) + (conflict * 2.0)
        
        # Pendulum physics step
        theta_accel = (self.gravity * math.sin(self.theta) + disturbance_force * math.cos(self.theta)) / self.length
        self.theta_dot += theta_accel * dt
        self.theta_dot *= max(0.0, 1.0 - self.damping * dt)
        self.theta += self.theta_dot * dt
        
        # Clamp theta to prevent physical flip-over
        self.theta = max(-math.pi / 2, min(math.pi / 2, self.theta))

    def apply_stabilizing_cortex_action(self, control_effort: float, dt: float = 0.05):
        # Stabilizing force applied by the prefrontal cortex (Mothership)
        stabilizing_accel = -control_effort / self.length
        self.theta_dot += stabilizing_accel * dt
        self.theta_dot *= max(0.0, 1.0 - self.damping * dt)
        self.theta += self.theta_dot * dt
        
class Mothership:
    def __init__(self, memory_engine: Any, emotion_handler: Any, scheduler: Any, execution_engine: Any = None):
        self.memory = memory_engine
        self.emotion = emotion_handler
        self.scheduler = scheduler
        self.execution_engine = execution_engine
        self.observation_factory = PerceptionObservationFactory()
        self.perception_formatter = PerceptionPromptFormatter()
        self.workspace_ctx = None
        
        # Instability regulator
        self.stability_regulator = CognitiveStabilityRegulator()
        self.pheromone_map: Dict[str, float] = {} # Shared pheromone trails (confidence)
        self.evaporation_rate = 0.10
        self._dreaming_active = threading.Event()

    def set_workspace(self, path: str) -> None:
        """Sets the active workspace path for local codebase inspection."""
        from src.input.hook.workspace import WorkspaceContext
        self.workspace_ctx = WorkspaceContext(path)
        logger.info(f"Workspace set to: {path}")

    def enter_dream_state(self) -> None:
        """Triggers the background dreaming/sleeping loop when idle."""
        if self._dreaming_active.is_set():
            return
        self._dreaming_active.set()
        self._dream_thread = threading.Thread(target=self._dream_loop, daemon=True)
        self._dream_thread.start()
        logger.info("[Mothership] Background dream/sleep cycle initiated.")

    def _dream_loop(self) -> None:
        """Continuous sleeping/consolidation loop run on a background daemon thread."""
        while self._dreaming_active.is_set():
            try:
                if hasattr(self.memory, "sleep"):
                    logger.debug("[Mothership] Executing memory sleep cycle.")
                    self.memory.sleep()
            except Exception as e:
                logger.error(f"[Mothership] Error in background dream cycle: {e}")
            
            # Sleep in tiny slices to allow immediate interruption on user prompts
            for _ in range(50):
                if not self._dreaming_active.is_set():
                    break
                time.sleep(0.1)

    def exit_dream_state(self) -> None:
        """Suspends background dreaming instantly to handle incoming queries."""
        if not self._dreaming_active.is_set():
            return
        self._dreaming_active.clear()
        if hasattr(self, "_dream_thread"):
            self._dream_thread.join()
        logger.info("[Mothership] Awake. Background dream cycle stopped.")

    def arbitrate_columns(self, cycle: int) -> List[CorticalColumn]:
        instability = abs(self.stability_regulator.theta)
        
        # 1. Base allocation: standard analytical and creative columns
        columns = [
            AnalyticalColumn(1, self.memory, self.emotion, self.scheduler),
            CreativeColumn(2, self.memory, self.emotion, self.scheduler)
        ]
        
        # 2. Reactive control allocation: if unstable, spin up risk auditing and verification
        if instability > 0.15:
            logger.info("Instability alert %.3f -> scheduling RiskColumn.", instability)
            columns.append(RiskColumn(3, self.memory, self.emotion, self.scheduler))
            
        if instability > 0.35:
            logger.info("Critical instability %.3f -> scheduling VerificationColumn.", instability)
            columns.append(VerificationColumn(4, self.memory, self.emotion, self.scheduler))
            
        # Set workspace path on all columns
        if self.workspace_ctx:
            for col in columns:
                col.workspace_dir = str(self.workspace_ctx.root_path)
            
        return columns

    def solve_problem(
        self,
        problem: str,
        max_cycles: int = 4,
    ) -> MothershipResponseDTO:
        # Guarantee dreaming is suspended before query execution
        self.exit_dream_state()

        if hasattr(self.memory, "clear_trajectory"):
            self.memory.clear_trajectory()

        working_thoughts: List[ThoughtDTO] = []
        seen_thought_texts: set[str] = set()
        best_overall_result = None
        cycles_used = 0
        errors: List[BrainErrorDTO] = []

        sensory_input = self._capture_perception_bundle(problem)

        for cycle in range(max_cycles):
            cycles_used = cycle + 1
            # 1. Prefrontal scheduling
            columns = self.arbitrate_columns(cycle)
            logger.info("Cycle %s active specialist columns=%s.", cycle + 1, len(columns))

            # 2. Execute columns and collect results
            cycle_results: List[NodeReasoningResultDTO] = []
            for col in columns:
                try:
                    start = time.perf_counter()
                    res = col.activate(sensory_input, working_thoughts)
                    duration_ms = (time.perf_counter() - start) * 1000.0
                    logger.info(
                        "Column %s completed in %.1fms with confidence %.2f.",
                        col.column_id,
                        duration_ms,
                        res.confidence,
                    )
                    cycle_results.append(res)
                    errors.extend(res.errors)
                except Exception as exc:
                    error = BrainErrorDTO(
                        component=f"column:{col.column_id}",
                        message=str(exc),
                        recoverable=True,
                    )
                    logger.exception("Column %s failed.", col.column_id)
                    errors.append(error)

            # 3. Calculate cognitive feedback signals for the stability regulator
            # Check average confidence of columns
            valid_results = [res for res in cycle_results if res.confidence > 0.0]
            uncertainty = 1.0 - self._average([res.confidence for res in valid_results]) if valid_results else 1.0
            
            # Stress signal from the emotional orchestrator
            stress = 0.2
            if hasattr(self.emotion, "current_homeostasis"):
                stress_val = getattr(self.emotion.current_homeostasis(), "stress", None)
                if stress_val is not None:
                    stress = stress_val
                
            # Conflict: variance/disagreement among column decisions
            decisions = [res.decision for res in cycle_results if res.decision]
            conflict = self._decision_conflict(decisions)

            # Perturb the pendulum system with these cognitive states
            self.stability_regulator.apply_cognitive_perturbations(uncertainty, stress, conflict)
            logger.info(
                "Cognitive feedback uncertainty=%.2f conflict=%.2f instability=%.4f.",
                uncertainty,
                conflict,
                self.stability_regulator.theta,
            )

            # 4. Apply prefrontal control effort to restore stability
            control_effort = 0.0
            for res in cycle_results:
                if res.decision and res.confidence > 0.70:
                    control_effort += 1.5 * (res.confidence - 0.5)
                    self.pheromone_map[res.decision] = self.pheromone_map.get(res.decision, 0.0) + res.confidence
                    if best_overall_result is None or res.confidence > best_overall_result.confidence:
                        best_overall_result = res
                        
            # Apply stabilizing action
            self.stability_regulator.apply_stabilizing_cortex_action(control_effort)

            # 5. Merge thought history and select paths using pheromone updates
            for res in cycle_results:
                for t in res.thought_history:
                    # Filter and add unique thoughts to global workspace
                    if t.raw_text not in seen_thought_texts:
                        working_thoughts.append(t)
                        seen_thought_texts.add(t.raw_text)

            # Evaporate pheromones
            for k in list(self.pheromone_map.keys()):
                self.pheromone_map[k] *= (1.0 - self.evaporation_rate)

            # 6. Check for convergence / early stopping criteria
            if abs(self.stability_regulator.theta) < 0.05 and best_overall_result and best_overall_result.confidence > 0.85:
                logger.info("Convergence detected in cycle %s. Stopping reasoning.", cycle + 1)
                break

        if hasattr(self.memory, "assign_credit_for_episode"):
            # Determine the final reward outcome (confidence of best result, or 0.0 if failed)
            reward = best_overall_result.confidence if best_overall_result else 0.0
            self.memory.assign_credit_for_episode(reward)

        return self._build_response(sensory_input, best_overall_result, working_thoughts, cycles_used, errors)

    def _capture_perception_bundle(self, problem: str) -> PerceptionBundleDTO:
        bundle = PerceptionBundleDTO(query=problem)
        if self.workspace_ctx is None:
            return bundle

        try:
            # 1. Perform workspace file listing
            files = self.workspace_ctx.list_dir()
            file_names = [f["name"] for f in files]
            dir_summary = f"Root directory files: {', '.join(file_names[:20])}"
            bundle.observations.append(
                self.observation_factory.from_capture("list_dir", dir_summary)
            )

            # 2. Grep search using query keywords
            keywords = [w.strip("?,.!") for w in problem.split() if len(w) > 3]
            grep_results = []
            for kw in keywords[:3]:  # search up to 3 keywords
                res = self.workspace_ctx.grep_search(kw)
                if res:
                    grep_results.extend(res[:5])
            
            if grep_results:
                grep_summary = "Relevant code search hits:\n" + "\n".join(
                    [f"- {r['file']}:{r['line_number']}: {r['content'][:100]}" for r in grep_results[:8]]
                )
                bundle.observations.append(
                    self.observation_factory.from_capture("grep_search", grep_summary)
                )
            
        except Exception as e:
            logger.error(f"Workspace perception capture error: {e}")
            bundle.observations.append(self.observation_factory.from_error("workspace", e))
            
        return bundle

    def _build_response(
        self,
        perception: PerceptionBundleDTO,
        best_result: Optional[NodeReasoningResultDTO],
        working_thoughts: List[ThoughtDTO],
        cycles_used: int,
        errors: Optional[List[BrainErrorDTO]] = None,
    ) -> MothershipResponseDTO:
        if best_result:
            decision = best_result.decision
            confidence = best_result.confidence
            goal_reached = best_result.goal_reached
            source_thoughts = best_result.thought_history
        else:
            decision = None
            confidence = 0.0
            goal_reached = False
            source_thoughts = working_thoughts

        reasoning_summary = self._summarize_reasoning(source_thoughts)
        if not decision and source_thoughts:
            latest_thought = source_thoughts[-1]
            decision = latest_thought.parsed_decision or latest_thought.thought_body
            confidence = latest_thought.confidence

        return MothershipResponseDTO(
            decision=decision,
            confidence=confidence,
            goal_reached=goal_reached,
            observations_used=self.perception_formatter.format_observation_names(perception.observations),
            reasoning_summary=reasoning_summary,
            cycles_used=cycles_used,
            errors=list(errors or []),
            metadata={
                "instability": abs(self.stability_regulator.theta),
                "pheromone_map": dict(self.pheromone_map),
            },
        )

    def _summarize_reasoning(self, thoughts: List[ThoughtDTO]) -> str:
        if not thoughts:
            return "No reasoning thoughts were generated."
        latest = thoughts[-1]
        summary = latest.thought_body or latest.raw_text
        return summary[:360]

    def _average(self, values: List[float]) -> float:
        return sum(values) / len(values)

    def _decision_conflict(self, decisions: List[str]) -> float:
        if len(decisions) <= 1:
            return 0.0
        counts: Dict[str, int] = {}
        for decision in decisions:
            normalized = decision.strip().lower()
            counts[normalized] = counts.get(normalized, 0) + 1
        majority = max(counts.values())
        return 1.0 - (majority / len(decisions))
