import math
from typing import Any, List, Dict, Iterable, Optional
from src.swarm.cells import AnalyticalColumn, CreativeColumn, RiskColumn, VerificationColumn, CorticalColumn
from src.brain.node.nodeDTOs import NodeReasoningResultDTO, ThoughtDTO
from src.body.perception.perceptionDTOs import PerceptionBundleDTO, PerceptionCaptureRequestDTO, PerceptionObservationDTO
from src.body.perception.perceptionFormatter import PerceptionObservationFactory, PerceptionPromptFormatter
from src.swarm.swarmDTOs import MothershipResponseDTO

class CognitiveStabilityRegulator:
    def __init__(self):
        self.theta = 0.0          # Angular deviation representing cognitive instability (rad)
        self.theta_dot = 0.0      # Rate of instability change (rad/s)
        self.gravity = 9.81
        self.length = 1.0

    def apply_cognitive_perturbations(self, uncertainty: float, stress: float, conflict: float, dt: float = 0.05):
        # Cognitive disturbances act as destabilizing forces pushing the pendulum away from theta=0
        disturbance_force = (uncertainty * 1.5) + (stress * 1.0) + (conflict * 2.0)
        
        # Pendulum physics step
        theta_accel = (self.gravity * math.sin(self.theta) + disturbance_force * math.cos(self.theta)) / self.length
        self.theta_dot += theta_accel * dt
        self.theta += self.theta_dot * dt
        
        # Clamp theta to prevent physical flip-over
        self.theta = max(-math.pi / 2, min(math.pi / 2, self.theta))

    def apply_stabilizing_cortex_action(self, control_effort: float, dt: float = 0.05):
        # Stabilizing force applied by the prefrontal cortex (Mothership)
        stabilizing_accel = -control_effort / self.length
        self.theta_dot += stabilizing_accel * dt
        self.theta += self.theta_dot * dt


class Mothership:
    def __init__(self, memory_engine: Any, emotion_handler: Any, scheduler: Any, execution_engine: Any = None):
        self.memory = memory_engine
        self.emotion = emotion_handler
        self.scheduler = scheduler
        self.execution_engine = execution_engine
        self.observation_factory = PerceptionObservationFactory()
        self.perception_formatter = PerceptionPromptFormatter()
        
        # Instability regulator
        self.stability_regulator = CognitiveStabilityRegulator()
        self.pheromone_map: Dict[str, float] = {} # Shared pheromone trails (confidence)
        self.evaporation_rate = 0.10

    def arbitrate_columns(self, cycle: int) -> List[CorticalColumn]:
        instability = abs(self.stability_regulator.theta)
        
        # 1. Base allocation: standard analytical and creative columns
        columns = [
            AnalyticalColumn(1, self.memory, self.emotion, self.scheduler),
            CreativeColumn(2, self.memory, self.emotion, self.scheduler)
        ]
        
        # 2. Reactive control allocation: if unstable, spin up risk auditing and verification
        if instability > 0.15:
            print(f"[Executive Cortex] Instability Alert ({instability:.3f}) -> Scheduling RiskColumn.")
            columns.append(RiskColumn(3, self.memory, self.emotion, self.scheduler))
            
        if instability > 0.35:
            print(f"[Executive Cortex] Critical Disagreements ({instability:.3f}) -> Scheduling VerificationColumn.")
            columns.append(VerificationColumn(4, self.memory, self.emotion, self.scheduler))
            
        return columns

    def solve_problem(
        self,
        problem: str,
        max_cycles: int = 4,
        devices_to_query: List[Any] = None
    ) -> MothershipResponseDTO:
        working_thoughts: List[ThoughtDTO] = []
        best_overall_result = None
        cycles_used = 0

        sensory_input = self._capture_perception_bundle(problem, devices_to_query)

        for cycle in range(max_cycles):
            cycles_used = cycle + 1
            # 1. Prefrontal scheduling
            columns = self.arbitrate_columns(cycle)
            print(f"[Executive Cortex] Cycle {cycle+1}: Active Specialist Columns = {len(columns)}")

            # 2. Execute columns and collect results
            cycle_results: List[NodeReasoningResultDTO] = []
            for col in columns:
                res = col.activate(sensory_input, working_thoughts)
                cycle_results.append(res)

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
            conflict = 1.0 - (len(set(decisions)) / len(decisions)) if decisions else 0.0

            # Perturb the pendulum system with these cognitive states
            self.stability_regulator.apply_cognitive_perturbations(uncertainty, stress, conflict)
            print(f"[Executive Cortex] Cognitive Feedback: Uncertainty={uncertainty:.2f}, Conflict={conflict:.2f} -> Instability Angle={self.stability_regulator.theta:.4f}")

            # 4. Apply prefrontal control effort to restore stability
            control_effort = 0.0
            for res in cycle_results:
                if res.decision and res.confidence > 0.70:
                    control_effort += 1.5 * (res.confidence - 0.5)
                    if best_overall_result is None or res.confidence > best_overall_result.confidence:
                        best_overall_result = res
                        
            # Apply stabilizing action
            self.stability_regulator.apply_stabilizing_cortex_action(control_effort)

            # 5. Merge thought history and select paths using pheromone updates
            for res in cycle_results:
                for t in res.thought_history:
                    # Filter and add unique thoughts to global workspace
                    if t.raw_text not in [w.raw_text for w in working_thoughts]:
                        working_thoughts.append(t)

            # Evaporate pheromones
            for k in list(self.pheromone_map.keys()):
                self.pheromone_map[k] *= (1.0 - self.evaporation_rate)

            # 6. Check for convergence / early stopping criteria
            if abs(self.stability_regulator.theta) < 0.05 and best_overall_result and best_overall_result.confidence > 0.85:
                print(f"[Executive Cortex] Convergence Detected in cycle {cycle+1}. Stopping reasoning.")
                break

        return self._build_response(sensory_input, best_overall_result, working_thoughts, cycles_used)

    def _capture_perception_bundle(
        self,
        problem: str,
        devices_to_query: Iterable[Any] = None
    ) -> PerceptionBundleDTO:
        bundle = PerceptionBundleDTO(query=problem)
        if not self.execution_engine or not devices_to_query:
            return bundle

        for request in self._normalize_capture_requests(devices_to_query):
            observation = self._capture_observation(request)
            bundle.observations.append(observation)
        return bundle

    def _capture_observation(self, request: PerceptionCaptureRequestDTO) -> PerceptionObservationDTO:
        if hasattr(self.execution_engine, "capture_observation"):
            return self.execution_engine.capture_observation(request.device, **request.arguments)
        try:
            payload = self.execution_engine.capture(request.device, **request.arguments)
            return self.observation_factory.from_capture(request.device, payload)
        except Exception as error:
            return self.observation_factory.from_error(request.device, error)

    def _normalize_capture_requests(self, devices_to_query: Iterable[Any]) -> List[PerceptionCaptureRequestDTO]:
        requests: List[PerceptionCaptureRequestDTO] = []
        for item in devices_to_query:
            if isinstance(item, PerceptionCaptureRequestDTO):
                requests.append(item)
            elif isinstance(item, str):
                requests.append(PerceptionCaptureRequestDTO(device=item))
            elif isinstance(item, dict):
                device = item.get("device") or item.get("name")
                if not device:
                    raise ValueError("Perception capture request dictionaries require a 'device' or 'name' key.")
                arguments = item.get("arguments") or item.get("kwargs") or {}
                requests.append(PerceptionCaptureRequestDTO(device=device, arguments=dict(arguments)))
            else:
                raise TypeError(f"Unsupported perception capture request: {type(item).__name__}")
        return requests

    def _build_response(
        self,
        perception: PerceptionBundleDTO,
        best_result: Optional[NodeReasoningResultDTO],
        working_thoughts: List[ThoughtDTO],
        cycles_used: int
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
            decision = source_thoughts[-1].thought_body

        return MothershipResponseDTO(
            decision=decision,
            confidence=confidence,
            goal_reached=goal_reached,
            observations_used=self.perception_formatter.format_observation_names(perception.observations),
            reasoning_summary=reasoning_summary,
            cycles_used=cycles_used,
        )

    def _summarize_reasoning(self, thoughts: List[ThoughtDTO]) -> str:
        if not thoughts:
            return "No reasoning thoughts were generated."
        latest = thoughts[-1]
        summary = latest.thought_body or latest.raw_text
        return summary[:360]

    def _average(self, values: List[float]) -> float:
        return sum(values) / len(values)
