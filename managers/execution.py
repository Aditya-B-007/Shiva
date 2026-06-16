import torch
from pydantic import BaseModel
from .runtime import runtime_manager

class ExecutionDirective(BaseModel):
    model_id: str
    directive: str
    user_id: str = "root_user"
    session_id: str = "default_terminal_session"

class ExecutionManager:
    def __init__(self):
        pass

    def process_directive(self, request: ExecutionDirective) -> dict:
        directive = request.directive
        context = runtime_manager.sql_memory.retrieve_context(
            user_id=request.user_id,
            session_id=request.session_id,
            current_query=directive
        )
        runtime_manager.sql_memory.save_chat_message(
            session_id=request.session_id,
            role="user",
            content=directive
        )

        facts_str = " | ".join(context["relevant_facts"]) if context["relevant_facts"] else "No prior facts."
        chat_str = " | ".join([f"{msg['role']}: {msg['content']}" for msg in context["recent_chat_history"]])
        
        augmented_directive = f"[FACTS: {facts_str}] [HISTORY: {chat_str}] [COMMAND: {directive}]"

        runtime_manager.emotional_core.update_homeostasis(action_impact=0.05, environment_surprise=0.1)
        
        try:
            z = runtime_manager.backbone.forward_pass(augmented_directive, runtime_manager.device)
            valence = runtime_manager.emotional_core.get_valence(z.mean(dim=1))
            
            if valence.mean().item() > 0:
                runtime_manager.emotional_core.mood_swing("Alert", "Processed complex structural directive.")
            else:
                runtime_manager.emotional_core.mood_swing("Calm", "Processed routine background directive.")
            
            final_action, final_log_prob, blending_gate = runtime_manager.policy.get_action(z)
            
            summary = f"Execution successful. Active Valence: {valence.mean().item():.4f}. Gate Blended at {blending_gate.mean().item():.2f}"
            runtime_manager.sql_memory.save_chat_message(
                session_id=request.session_id,
                role="assistant",
                content=summary
            )

        except Exception as e:
            summary = f"Execution halted. System fault: {str(e)}"
            runtime_manager.sql_memory.save_chat_message(
                session_id=request.session_id,
                role="system",
                content=summary
            )

        return {
            "status": "success",
            "execution_summary": summary,
            "cognitive_state": runtime_manager.get_cognitive_state()
        }

execution_manager = ExecutionManager()
