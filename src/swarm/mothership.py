import math
import logging
import time
import threading
from typing import Any, List, Dict, Iterable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from src.brain.transformer.Encoder import Encoder
from src.swarm.cells import AnalyticalColumn, CreativeColumn, RiskColumn, VerificationColumn, CorticalColumn
from src.transferDTO import (
    BrainErrorDTO,
    NodeReasoningResultDTO,
    ThoughtDTO,
    PerceptionBundleDTO,
    PerceptionCaptureRequestDTO,
    PerceptionObservationDTO,
    MothershipResponseDTO,
)
from src.input.hook.perception import (
    PerceptionObservationFactory,
    PerceptionPromptFormatter,
)

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

        # Neuro-symbolic Value network for TD learning
        self.encoder = Encoder()
        self._goal_projector = None
        self._state_projector = None
        self._value_network = None
        self._value_optimizer = None

    def _encode_text(self, text: str) -> torch.Tensor:
        from src.transferDTO import EncoderInputDTO
        device = self.encoder.device
        dto = EncoderInputDTO(text=text)
        try:
            enc_in = self.encoder.input(dto)
            enc_out = self.encoder.process(enc_in)
            pooler_out = enc_out.pooler_output_pt
            if pooler_out is None:
                pooler_out = enc_out.last_hidden_state_pt[:, 0, :]
            return pooler_out
        except Exception as e:
            logger.error(f"Error encoding text in Mothership: {e}")
            return torch.zeros((1, 1024), device=device)

    def set_workspace(self, path: str) -> None:
        from src.input.hook.workspace import WorkspaceContext
        self.workspace_ctx = WorkspaceContext(path)
        logger.info(f"Workspace set to: {path}")

    def enter_dream_state(self) -> None:
        if self._dreaming_active.is_set():
            return
        self._dreaming_active.set()
        self._dream_thread = threading.Thread(target=self._dream_loop, daemon=True)
        self._dream_thread.start()
        logger.info("[Mothership] Background dream/sleep cycle initiated.")

    def _dream_loop(self) -> None:
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
        if not self._dreaming_active.is_set():
            return
        self._dreaming_active.clear()
        if hasattr(self, "_dream_thread"):
            self._dream_thread.join()
        logger.info("[Mothership] Awake. Background dream cycle stopped.")

    def arbitrate_columns(self, cycle: int) -> List[CorticalColumn]:
        instability = abs(self.stability_regulator.theta)
        columns = [
            AnalyticalColumn(1, self.memory, self.emotion, self.scheduler),
            CreativeColumn(2, self.memory, self.emotion, self.scheduler)
        ]
        if instability > 0.15:
            logger.info("Instability alert %.3f -> scheduling RiskColumn.", instability)
            columns.append(RiskColumn(3, self.memory, self.emotion, self.scheduler))
            
        if instability > 0.35:
            logger.info("Critical instability %.3f -> scheduling VerificationColumn.", instability)
            columns.append(VerificationColumn(4, self.memory, self.emotion, self.scheduler))
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

        # Lazily initialize goal and state projectors and value network
        device = self.encoder.device
        if self._goal_projector is None:
            self._goal_projector = nn.Linear(1024, 2048).to(device)
            nn.init.normal_(self._goal_projector.weight, std=0.02)
            nn.init.zeros_(self._goal_projector.bias)
        if self._state_projector is None:
            self._state_projector = nn.Linear(1024, 2048).to(device)
            nn.init.normal_(self._state_projector.weight, std=0.02)
            nn.init.zeros_(self._state_projector.bias)
        if self._value_network is None:
            self._value_network = nn.Sequential(
                nn.Linear(4096, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ).to(device)
            self._value_optimizer = optim.Adam(self._value_network.parameters(), lr=1e-3)

        # 1. Encode goal vector
        goal_pooler = self._encode_text(problem)
        with torch.no_grad():
            goal_emb_t = self._goal_projector(goal_pooler)
            goal_emb = goal_emb_t.squeeze(0).cpu().numpy()

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

            # 2. Encode current state vector
            state_text = f"Perception: {problem} | Thoughts: " + " ".join([t.thought_body for t in working_thoughts])
            state_pooler = self._encode_text(state_text)
            with torch.no_grad():
                state_emb_t = self._state_projector(state_pooler)
                state_emb = state_emb_t.squeeze(0).cpu().numpy()

            # Prefrontal scheduling
            columns = self.arbitrate_columns(cycle)
            logger.info("Cycle %s active specialist columns=%s.", cycle + 1, len(columns))

            # Set goal-conditioned embeddings on columns
            for col in columns:
                col.current_goal_emb = goal_emb
                col.current_state_emb = state_emb

            # Execute columns and collect results
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

            # Calculate average step reward of cycle columns
            step_rewards = [getattr(col, "last_step_reward", 0.0) for col in columns]
            step_reward = float(np.mean(step_rewards)) if step_rewards else 0.0

            # 3. Calculate cognitive feedback signals for the stability regulator
            valid_results = [res for res in cycle_results if res.confidence > 0.0]
            uncertainty = 1.0 - self._average([res.confidence for res in valid_results]) if valid_results else 1.0
            
            stress = 0.2
            if hasattr(self.emotion, "current_homeostasis"):
                stress_val = getattr(self.emotion.current_homeostasis(), "stress", None)
                if stress_val is not None:
                    stress = stress_val
                
            decisions = [res.decision for res in cycle_results if res.decision]
            conflict = self._decision_conflict(decisions)

            # Compute next state embedding
            next_state_text = f"Perception: {problem} | Thoughts: " + " ".join([t.thought_body for t in working_thoughts])
            next_state_pooler = self._encode_text(next_state_text)
            with torch.no_grad():
                next_state_emb_t = self._state_projector(next_state_pooler)
                next_state_emb = next_state_emb_t.squeeze(0).cpu().numpy()

            # Compute TD Error (RPE)
            state_goal_t = torch.cat([state_emb_t, goal_emb_t], dim=-1)
            next_state_goal_t = torch.cat([next_state_emb_t, goal_emb_t], dim=-1)
            
            with torch.no_grad():
                v_s = self._value_network(state_goal_t).item()
                v_s_next = self._value_network(next_state_goal_t).item()
                
            gamma = 0.99
            rpe = step_reward + gamma * v_s_next - v_s

            # Train value network parameters using TD target MSE loss
            td_target = torch.tensor([[step_reward + gamma * v_s_next]], device=device)
            predicted_v = self._value_network(state_goal_t)
            v_loss = F.mse_loss(predicted_v, td_target)
            
            self._value_optimizer.zero_grad()
            v_loss.backward()
            self._value_optimizer.step()
            
            logger.info(
                "[Mothership TD] Cycle %s step_reward=%.2f V(s)=%.2f V(s')=%.2f RPE=%.2f loss=%.4f",
                cycle + 1,
                step_reward,
                v_s,
                v_s_next,
                rpe,
                v_loss.item()
            )

            # Perturb the pendulum system with cognitive state + RPE
            # Negative RPE increases uncertainty/instability; positive stabilizes
            perturbation = float(-rpe * 1.5)
            self.stability_regulator.apply_cognitive_perturbations(
                uncertainty=max(0.0, uncertainty + perturbation),
                stress=stress,
                conflict=conflict
            )
            logger.info(
                "Cognitive feedback uncertainty=%.2f conflict=%.2f instability=%.4f.",
                uncertainty,
                conflict,
                self.stability_regulator.theta,
            )

            # Apply prefrontal control effort to restore stability
            control_effort = 0.0
            for res in cycle_results:
                if res.decision and res.confidence > 0.70:
                    control_effort += 1.5 * (res.confidence - 0.5)
                    self.pheromone_map[res.decision] = self.pheromone_map.get(res.decision, 0.0) + res.confidence
                    if best_overall_result is None or res.confidence > best_overall_result.confidence:
                        best_overall_result = res
                        
            # Apply stabilizing action
            self.stability_regulator.apply_stabilizing_cortex_action(control_effort)

            # Merge thought history and select paths using pheromone updates
            for res in cycle_results:
                for t in res.thought_history:
                    if t.raw_text not in seen_thought_texts:
                        working_thoughts.append(t)
                        seen_thought_texts.add(t.raw_text)

            # Evaporate pheromones
            for k in list(self.pheromone_map.keys()):
                self.pheromone_map[k] *= (1.0 - self.evaporation_rate)

            # Check for convergence / early stopping criteria
            if abs(self.stability_regulator.theta) < 0.05 and best_overall_result and best_overall_result.confidence > 0.85:
                logger.info("Convergence detected in cycle %s. Stopping reasoning.", cycle + 1)
                break

        if hasattr(self.memory, "assign_credit_for_episode"):
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
