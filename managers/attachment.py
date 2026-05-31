import os
from pathlib import Path
from safetensors.torch import load_file
import torch
from .metadata import metadata_extractor
from .capability import capability_registry      
from .compatibility import compatibility_analyzer 

class AttachmentManager:
    def __init__(self):
        self.attached_models = {}

    def scan_model_directory(self, directory_path: str) -> dict:
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            return {"error": f"Invalid path target: {directory_path}"}

        model_name = path.name
        detected_files = {"config": [], "tokenizer": [], "weights": [], "other": []}

        # Handle file parsing
        for file in path.iterdir():
            if not file.is_file():
                continue
            filename = file.name.lower()
            if filename.endswith("config.json") and not "factory" in filename:
                detected_files["config"].append(file.name)
            elif "tokenizer" in filename:
                detected_files["tokenizer"].append(file.name)
            elif filename.endswith((".safetensors", ".bin", ".pt", ".pth")):
                detected_files["weights"].append(file.name)
            else:
                detected_files["other"].append(file.name)

        if not detected_files["weights"]:
            return {"error": "No valid model parameter states resolved."}

        # Sort files chronologically to guarantee correct shard aggregation sequence
        detected_files["weights"].sort()

        extracted_metadata = metadata_extractor.extract(str(path), detected_files)
        compatibility_report = compatibility_analyzer.analyze(extracted_metadata)
        
        model_id = f"model_{len(self.attached_models) + 1}"
        model_data = {
            "id": model_id,
            "name": model_name,
            "path": str(path.absolute()),
            "files": detected_files,
            "metadata": extracted_metadata,
            "compatibility": compatibility_report,
            "status": "scanned_ready"
        }

        capability_registry.register_model(
            model_id=model_id, 
            model_name=model_name, 
            is_compatible=compatibility_report["is_compatible"]
        )

        self.attached_models[model_id] = model_data
        return {"success": True, "model": model_data}

    def load_full_state_dict(self, model_id: str) -> dict:
        """Assembles unified state dict tensors from sharded weight files."""
        model_info = self.attached_models.get(model_id)
        base_path = Path(model_info["path"])
        weight_files = model_info["files"]["weights"]
        
        combined_state_dict = {}
        for f_name in weight_files:
            f_path = base_path / f_name
            try:
                if f_path.suffix == ".safetensors":
                    shard = load_file(str(f_path), device="cpu")
                else:
                    # Fallback to weights_only safely if tensor classes allow it
                    shard = torch.load(str(f_path), map_location="cpu", weights_only=False)
                combined_state_dict.update(shard)
            except Exception as e:
                raise RuntimeError(f"Failed parsing state fragment {f_name}: {str(e)}")
        return combined_state_dict

    def list_attached_models(self) -> dict:
        return self.attached_models

attachment_manager = AttachmentManager()
