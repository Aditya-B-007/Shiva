import torch
from pydantic import BaseModel
from .runtime import runtime_manager

class ExecutionDirective(BaseModel):
    model_id: str
    directive: str

class ExecutionManager:
    def __init__(self):
        self.execution_history = []

    def process_directive(self, request: ExecutionDirective) -> dict:
        directive = request.directive

        runtime_manager.emotional_core.update_homeostasis(action_impact=0.05, environment_surprise=0.1)
        try:
            # Pass text directive directly to the upgraded text embedding backbone
            z = runtime_manager.backbone.forward_pass(directive, runtime_manager.device)
            valence = runtime_manager.emotional_core.get_valence(z.mean(dim=1))
            
            if valence.mean().item() > 0:
                runtime_manager.emotional_core.mood_swing("Alert", "Processed complex structural directive.")
            else:
                runtime_manager.emotional_core.mood_swing("Calm", "Processed routine background directive.")
            
            summary = f"Real text embedding mapping successful. Active Valence: {valence.mean().item():.4f}."

        except Exception as e:
            summary = f"Execution halted. System fault: {str(e)}"

        return {
            "status": "success",
            "execution_summary": summary,
            "cognitive_state": runtime_manager.get_cognitive_state()
        }

execution_manager = ExecutionManager()

