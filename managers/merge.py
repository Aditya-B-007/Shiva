import torch
from pathlib import Path
from safetensors.torch import load_file
from core.merge_strategies import RapidFrankenmergeStrategy
from .runtime import runtime_manager
from .attachment import attachment_manager
from .merge import merge_manager


class FrankenMergeManager:
    def __init__(self):
        self.strategy = RapidFrankenmergeStrategy()

    def perform_merge(self, model_id: str) -> dict:
        model_info = attachment_manager.attached_models.get(model_id)
        if not model_info:
            return {"success": False, "error": "Model not found in registry."}
        
        if not model_info["compatibility"]["is_compatible"]:
            return {"success": False, "error": "Model architecture is incompatible."}
        base_path = Path(model_info["path"])
        weight_files = model_info["files"]["weights"]
        weight_path = base_path / weight_files[0]
        
        try:
            if weight_path.suffix == ".safetensors":
                ext_state_dict = load_file(weight_path)
            else:
                ext_state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
        except Exception as e:
            return {"success": False, "error": f"Failed to load weights: {str(e)}"}
        ext_config = {
            "num_heads": model_info["metadata"]["heads"],
            "hidden_size": model_info["metadata"]["hidden_size"]
        }

        try:
            new_state = self.strategy.merge(
                target_model=runtime_manager.policy, 
                ext_state_dict=ext_state_dict, 
                ext_config=ext_config
            )
            runtime_manager.policy.load_state_dict(new_state, strict=False)
            runtime_manager.emotional_core.update_homeostasis(action_impact=0.3, environment_surprise=0.8)
            runtime_manager.emotional_core.mood_swing("Alert", "Ingested massive external parameter distribution.")
            
            return {
                "success": True, 
                "message": "FrankenMerge successful. Weights compressed and ingested via SVD.",
                "cognitive_state": runtime_manager.get_cognitive_state()
            }

        except Exception as e:
            return {"success": False, "error": f"Merge failed during SVD/Compression: {str(e)}"}

merge_manager = FrankenMergeManager()
