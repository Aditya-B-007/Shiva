from abc import ABC, abstractmethod
from typing import Any, Dict, List
import torch
from src.transferDTO import ThoughtDTO, NodeReasoningResultDTO, PerceptionBundleDTO
from src.brain.node.scratchPad import ScratchPad
from src.brain.node.chainOfThought import ChainOfThought
from src.brain.node.nodeProcessingEngine import nodeProcessingEngine
from src.input.hook.perception import PerceptionPromptFormatter

from src.swarm.SwarmSAC import SwarmSAC

class CorticalColumn(ABC):
    def __init__(self, column_id: int, memory_engine: Any, emotion_handler: Any, scheduler: Any):
        """
        Base class representing a specialized cortical column in the swarm.
        """
        self.column_id = column_id
        self.memory = memory_engine
        self.emotion = emotion_handler
        self.scheduler = scheduler
        self.scratchpad = ScratchPad()
        self.chain = ChainOfThought()
        self.engine = nodeProcessingEngine(
            memory_engine=self.memory,
            emotion_handler=self.emotion,
            scratchpad=self.scratchpad,
            chain_of_thought=self.chain,
            reasoning_scheduler=self.scheduler
        )
        self.perception_formatter = PerceptionPromptFormatter()
        self.sac = SwarmSAC(device="cuda" if torch.cuda.is_available() else "cpu")
        self.last_step_reward = 0.0

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """
        Custom instructions defining the role and thinking perspective of this column.
        """
        pass

    @property
    @abstractmethod
    def search_policy(self) -> Dict[str, Any]:
        """
        Custom decoder settings (e.g. temperature, top_p, max_new_tokens).
        """
        pass

    def retrieve_context(self, perception: Any) -> List[Any]:
        """
        Define memory and emotion retrieval priorities specific to this column.
        """
        raw = self.memory.retrieve(self._memory_query(perception), limit=3)
        return list(getattr(raw, "memories", raw))

    def activate(self, perception: Any, working_history: List[ThoughtDTO] = None) -> NodeReasoningResultDTO:
        import torch
        import numpy as np

        # 1. Retrieve or initialize goal and state embeddings
        state_emb = getattr(self, "current_state_emb", None)
        if state_emb is None:
            state_emb = np.zeros(2048, dtype=np.float32)
        goal_emb = getattr(self, "current_goal_emb", None)
        if goal_emb is None:
            goal_emb = np.zeros(2048, dtype=np.float32)
            
        # Concatenate into goal-conditioned state vector
        state_goal_vector = np.concatenate([state_emb, goal_emb])
        state_goal_tensor = torch.FloatTensor(state_goal_vector).to(self.sac.device)
        
        # 2. ACTOR: Sample latent action projection vector from SAC Actor
        with torch.no_grad():
            latent_action, log_prob = self.sac.actor.sample(state_goal_tensor)
            latent_action_np = latent_action.cpu().numpy()
            
        # 3. GENERATION: Pass latent action to the Decoder to guide the reasoning step
        formatted_perception = self._format_for_reasoning(perception)
        kwargs = dict(self.search_policy)
        kwargs["latent_action"] = latent_action
        if hasattr(self, "workspace_dir") and self.workspace_dir:
            kwargs["workspace_dir"] = self.workspace_dir
            
        res = self.engine.process(
            formatted_perception,
            seed_thoughts=working_history or [],
            memories=self.retrieve_context(perception),
            decoder_kwargs=kwargs,
        )
        
        # 4. CRITIC: Validate the generated action
        decision_text = res.decision or ""
        step_reward = 0.0
        
        if "```python" in decision_text:
            try:
                start_idx = decision_text.find("```python") + len("```python")
                end_idx = decision_text.find("```", start_idx)
                if end_idx != -1:
                    code_block = decision_text[start_idx:end_idx].strip()
                    
                    # Generate validation test script using the Decoder
                    validation_prompt = (
                        f"You are the Swarm Critic validation engine. "
                        f"Write a Python 3 assertion unit test script to validate the correctness of the following script:\n"
                        f"```python\n{code_block}\n```\n"
                        f"The validation script must run assertions and print 'Validation Passed' if correct. "
                        f"Ensure you only output the validation script inside a single ```python block."
                    )
                    
                    with self.scheduler.acquire_decoder() as decoder:
                        from src.transferDTO import ReasoningContextDTO
                        mock_ctx = ReasoningContextDTO(perception=validation_prompt)
                        val_thought = decoder.generateDecision(mock_ctx)
                        val_decision = val_thought.parsed_decision or val_thought.thought_body or ""
                        
                        if "```python" in val_decision:
                            v_start = val_decision.find("```python") + len("```python")
                            v_end = val_decision.find("```", v_start)
                            if v_end != -1:
                                val_code = val_decision[v_start:v_end].strip()
                                # Run validation script in the sandbox
                                val_result = decoder.execute_sandbox_script(val_code, getattr(self, "workspace_dir", None))
                                if "Sandbox execution error" in val_result or "AssertionError" in val_result:
                                    step_reward = -0.5
                                    res.decision += f"\n\n[Swarm Critic]: Validation Failed:\n{val_result}"
                                else:
                                    step_reward = 0.5
                                    res.decision += f"\n\n[Swarm Critic]: Validation Passed:\n{val_result}"
            except Exception:
                step_reward = -0.2
        
        # 5. TD Learning reinforcement step
        if res.errors:
            step_reward -= 0.5
        else:
            step_reward += 0.2
            
        step_reward = float(np.clip(step_reward, -1.0, 1.0))
        self.last_step_reward = step_reward
        
        # Store transition experience in Replay Buffer
        next_state_emb = np.zeros(2048, dtype=np.float32)
        next_state_goal_vector = np.concatenate([next_state_emb, goal_emb])
        
        self.sac.add_experience(
            state=state_goal_vector,
            action=latent_action_np[0] if latent_action_np.ndim > 1 else latent_action_np,
            reward=step_reward,
            next_state=next_state_goal_vector,
            done=res.goal_reached
        )
        
        # Train networks
        self.sac.update_parameters(batch_size=min(len(self.sac.buffer), 16))
        
        return res

    def _format_for_reasoning(self, perception: Any) -> str:
        if isinstance(perception, PerceptionBundleDTO):
            formatted_perception = self.perception_formatter.format_bundle(perception)
        else:
            formatted_perception = str(perception)

        return (
            f"Specialty Instruction: {self.system_prompt}\n"
            f"{formatted_perception}"
        )

    def _memory_query(self, perception: Any) -> str:
        if isinstance(perception, PerceptionBundleDTO):
            return self.perception_formatter.format_bundle(perception)
        return str(perception)



class AnalyticalColumn(CorticalColumn):
    @property
    def system_prompt(self) -> str:
        return "You are an Analytical Cortical Column. Focus purely on logic, evidence, and mathematical correctness."
        
    @property
    def search_policy(self) -> Dict[str, Any]:
        return {"temperature": 0.1, "top_p": 0.85, "max_new_tokens": 32768}


class CreativeColumn(CorticalColumn):
    @property
    def system_prompt(self) -> str:
        return "You are a Creative Cortical Column. Brainstorm analogies, lateral hypotheses, and novel connection paths."
        
    @property
    def search_policy(self) -> Dict[str, Any]:
        return {"temperature": 0.85, "top_p": 0.95, "max_new_tokens": 32768}


class RiskColumn(CorticalColumn):
    @property
    def system_prompt(self) -> str:
        return "You are a Risk & Safety Column. Focus on identifying failure modes, edge-case risks, and system instability."
        
    @property
    def search_policy(self) -> Dict[str, Any]:
        return {"temperature": 0.2, "top_p": 0.90, "max_new_tokens": 32768}


class VerificationColumn(CorticalColumn):
    @property
    def system_prompt(self) -> str:
        return "You are a Verification Column. Audit the thoughts generated by other columns for contradictions and factual errors."
        
    @property
    def search_policy(self) -> Dict[str, Any]:
        return {"temperature": 0.05, "top_p": 0.80, "max_new_tokens": 32768}
