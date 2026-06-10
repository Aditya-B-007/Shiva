import json
from pathlib import Path

class MetadataExtractor:
    def __init__(self):
        pass

    def extract(self, directory_path: str, detected_files: dict) -> dict:
        path = Path(directory_path)
        metadata = {
            "architecture": "unknown",
            "layers": 0,
            "hidden_size": 0,
            "heads": 0,
            "vocab_size": 0,
            "context_window": 0
        }

        config_files = detected_files.get("config", [])
        if not config_files:
            return metadata
        config_path = path / config_files[0]
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
            archs = config_data.get("architectures", ["unknown"])
            metadata["architecture"] = archs[0] if archs else "unknown"
            metadata["layers"] = config_data.get("num_hidden_layers", 0)
            metadata["hidden_size"] = config_data.get("hidden_size", 0)
            metadata["heads"] = config_data.get("num_attention_heads", 0)
            metadata["vocab_size"] = config_data.get("vocab_size", 0)
            metadata["context_window"] = config_data.get("max_position_embeddings", 0)
            
            if config_data.get("is_patched_ollama"):
                metadata["parameters"] = "Ollama Blob State (Auto-fitted)"

        except Exception as e:
            metadata["extraction_error"] = str(e)

        return metadata

metadata_extractor = MetadataExtractor()
