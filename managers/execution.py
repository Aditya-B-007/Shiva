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
            dummy_seq_length = len(directive.split())
            dummy_input = torch.randn(1, dummy_seq_length, 256).to(runtime_manager.device)
            
            with torch.no_grad():
                z = runtime_manager.backbone.forward_pass(dummy_input)
                valence = runtime_manager.emotional_core.get_valence(z.mean(dim=1))
                
                if valence.mean().item() > 0:
                    runtime_manager.emotional_core.mood_swing("Alert", "Processed complex structural directive.")
                else:
                    runtime_manager.emotional_core.mood_swing("Calm", "Processed routine background directive.")
            summary = f"Latent mapping successful. Valence variance: {valence.mean().item():.4f}."

        except Exception as e:
            summary = f"Execution halted. System fault: {str(e)}"

        return {
            "status": "success",
            "execution_summary": summary,
            "cognitive_state": runtime_manager.get_cognitive_state()
        }

execution_manager = ExecutionManager()
