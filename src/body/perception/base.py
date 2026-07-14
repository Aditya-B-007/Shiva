from abc import ABC, abstractmethod
from typing import Any, Dict

class PerceptionDevice(ABC):
    """
    Base interface representing a sensory organ/perception device in the Shiva body.
    Perception devices only collect raw information from the external environment.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name of the perception device.
        """
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what sensory inputs this device captures.
        """
        pass

    @property
    @abstractmethod
    def parameter_definitions(self) -> Dict[str, Any]:
        """
        Expected input parameters for capturing data (e.g. format, timeout).
        """
        pass

    @abstractmethod
    def capture(self, **kwargs: Any) -> Any:
        """
        Captures raw sensor observations.
        Returns:
            Raw data (bytes, string, numeric arrays, dictionary, etc.) depending on the device.
            Should never return natural language reasoning, explanations, or user responses.
        """
        pass
