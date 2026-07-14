import time
from typing import Any, Dict
from ..registry.PerceptionRegistry import PerceptionRegistry

class ExecutionEngine:
    """
    The engine responsible for orchestrating sensory capture.
    It dispatches capture commands to registered perception devices and returns raw observations.
    """

    def __init__(self, registry: PerceptionRegistry) -> None:
        self.registry = registry

    def capture(self, device_name: str, **kwargs: Any) -> Any:
        """
        Dispatches perception requests to the specified device and returns raw observations.
        """
        device = self.registry.get_device(device_name)
        if not device:
            raise ValueError(f"Perception device '{device_name}' not found in registry.")

        self._validate_arguments(device, kwargs)
        return device.capture(**kwargs)

    def _validate_arguments(self, device: Any, arguments: Dict[str, Any]) -> None:
        defs = device.parameter_definitions
        for param_name, param_info in defs.items():
            is_required = param_info.get("required", True)
            if is_required and param_name not in arguments:
                raise ValueError(f"Missing required parameter '{param_name}' for perception device '{device.name}'")

            if param_name in arguments:
                val = arguments[param_name]
                expected_type = param_info.get("type", "string")
                if expected_type == "integer" and not isinstance(val, int):
                    if isinstance(val, str) and val.isdigit():
                        arguments[param_name] = int(val)
                    else:
                        raise TypeError(f"Parameter '{param_name}' must be an integer, got {type(val).__name__}")
                elif expected_type == "string" and not isinstance(val, str):
                    arguments[param_name] = str(val)
