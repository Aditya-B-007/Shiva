import os
from pathlib import Path
from managers.metadata import metadata_extractor 

class AttachmentManager:
    def __init__(self):
        self.attached_models = {}

    def scan_model_directory(self, directory_path: str) -> dict:
        path = Path(directory_path)
        
        if not path.exists():
            return {"error": f"Directory not found: {directory_path}"}
        if not path.is_dir():
            return {"error": f"Path is not a directory: {directory_path}"}

        model_name = path.name
        detected_files = {
            "config": [],
            "tokenizer": [],
            "weights": [],
            "other": []
        }

        for file in path.iterdir():
            if not file.is_file():
                continue
            filename = file.name.lower()
            if filename.endswith("config.json"):
                detected_files["config"].append(file.name)
            elif "tokenizer" in filename:
                detected_files["tokenizer"].append(file.name)
            elif filename.endswith((".safetensors", ".bin", ".pt", ".pth")):
                detected_files["weights"].append(file.name)
            else:
                detected_files["other"].append(file.name)

        if not detected_files["weights"]:
            return {"error": "No model weight files (.safetensors, .bin, .pt) found in directory."}

        extracted_metadata = metadata_extractor.extract(str(path), detected_files)
        compatibility_report = compatibility_analyzer.analyze(extracted_metadata)
        model_data = {
            "id": f"model_{len(self.attached_models) + 1}",
            "name": model_name,
            "path": str(path.absolute()),
            "files": detected_files,
            "metadata": extracted_metadata,
            "compatibility":compatibility_report,
            "status": "scanned_ready"
        }

        capability_registry.register_model(
            model_id=model_data["id"], 
            model_name=model_name, 
            is_compatible=compatibility_report["is_compatible"]
        )

        self.attached_models[model_data["id"]] = model_data
        return {"success": True, "model": model_data}

    def list_attached_models(self) -> dict:
        return self.attached_models

attachment_manager = AttachmentManager()
