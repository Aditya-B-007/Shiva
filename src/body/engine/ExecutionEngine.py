import time
from typing import Any, Dict
from ..registry.PerceptionRegistry import PerceptionRegistry
from ..perception.perceptionDTOs import PerceptionObservationDTO
from ..perception.perceptionFormatter import PerceptionObservationFactory

class ExecutionEngine:
    """
    The engine responsible for orchestrating sensory capture.
    It dispatches capture commands to registered perception devices and returns raw observations.
    """

    def __init__(self, registry: PerceptionRegistry) -> None:
        self.registry = registry
        self.observation_factory = PerceptionObservationFactory()

    def capture(self, device_name: str, **kwargs: Any) -> Any:
        """
        Dispatches perception requests to the specified device and returns raw observations.
        """
        device = self.registry.get_device(device_name)
        if not device:
            raise ValueError(f"Perception device '{device_name}' not found in registry.")

        self._validate_arguments(device, kwargs)
        return device.capture(**kwargs)

    def capture_observation(self, device_name: str, **kwargs: Any) -> PerceptionObservationDTO:
        """
        Dispatches perception requests and returns a structured observation DTO.
        Raw capture remains available through capture() for existing integrations.
        """
        try:
            payload = self.capture(device_name, **kwargs)
            return self.observation_factory.from_capture(device_name, payload)
        except Exception as error:
            return self.observation_factory.from_error(device_name, error)

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
