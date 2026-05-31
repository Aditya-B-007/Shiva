class CapabilityRegistry:
    def __init__(self):
        self.capabilities = {
            "frankenmerge": {
                "type": "weight_ingestion",
                "strategy": "RapidFrankenmergeStrategy",
                "status": "ready"
            },
            "parasitic_extraction": {
                "type": "activation_interception",
                "strategy": "ProbeNetwork",
                "status": "ready"
            },
            "swarm_consensus": {
                "type": "global_workspace",
                "strategy": "CrossAttentionAggregator",
                "status": "ready"
            }
        }
        self.attached_models_manifest = {}

    def register_model(self, model_id: str, model_name: str, is_compatible: bool) -> None:
        self.attached_models_manifest[model_id] = {
            "name": model_name,
            "is_compatible": is_compatible,
            "ingestion_vector": "SVD_Head_Compress" if is_compatible else "Parasitic_Hook_Only"
        }

    def list_capabilities(self):
        return {
            "core_capabilities": self.capabilities,
            "registered_models": self.attached_models_manifest
        }

capability_registry = CapabilityRegistry()
