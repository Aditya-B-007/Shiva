class CompatibilityAnalyzer:
    def __init__(self, target_d_model: int = 512, target_heads: int = 8):
        self.target_d_model = target_d_model
        self.target_heads = target_heads

    def analyze(self, metadata: dict) -> dict:
        report = {
            "is_compatible": True,
            "overall_status": "Ready for Merge",
            "required_operations": [],
            "warnings": []
        }
        if metadata.get("architecture") == "unknown":
            report["is_compatible"] = False
            report["overall_status"] = "Incompatible: Unknown Architecture"
            report["warnings"].append("Cannot safely map unknown attention blocks.")
            return report
        ext_hidden = metadata.get("hidden_size", 0)
        if ext_hidden != self.target_d_model and ext_hidden > 0:
            report["required_operations"].append({
                "name": "SVD Dimension Fitting",
                "detail": f"Will compress {ext_hidden}d -> {self.target_d_model}d via truncated SVD."
            })
            report["overall_status"] = "SVD Fitting Required"
        ext_heads = metadata.get("heads", 0)
        if ext_heads != self.target_heads and ext_heads > 0:
            report["required_operations"].append({
                "name": "Attention Head Compression",
                "detail": f"Will bucket-average {ext_heads} source heads down to {self.target_heads} target heads."
            })
            report["overall_status"] = "Head Averaging Required"
        if len(report["required_operations"]) > 1:
            report["overall_status"] = "Aggressive SVD + Head Compression Required"
        elif len(report["required_operations"]) == 0:
            report["overall_status"] = "Native 1:1 Compatibility"

        return report

compatibility_analyzer = CompatibilityAnalyzer()
