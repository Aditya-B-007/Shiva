from typing import Dict, List, Optional
from ..perception.base import PerceptionDevice

class PerceptionRegistry:
    """
    Registry that dynamically holds and manages all available PerceptionDevice sensors.
    """

    def __init__(self) -> None:
        self._devices: Dict[str, PerceptionDevice] = {}

    def register(self, device: PerceptionDevice) -> None:
        if not device.name:
            raise ValueError("Perception device name cannot be empty")
        self._devices[device.name] = device

    def get_device(self, name: str) -> Optional[PerceptionDevice]:
        return self._devices.get(name)

    def get_all_devices(self) -> List[PerceptionDevice]:
        return list(self._devices.values())
