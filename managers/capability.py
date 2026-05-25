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
    #WE WILL BE INCREASING THE CAPABILITIES AS AND WHEN THEY ARE ADDED IN THE FUTURE.
    def list_capabilities(self):
        return self.capabilities

capability_registry = CapabilityRegistry()
