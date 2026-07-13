from abc import ABC, abstractmethod
from typing import Any, Dict
from ..models.ToolResult import ToolResult
from ..models.ToolMetadata import ToolMetadata

class Tool(ABC):
    """Abstract Base Class (interface) that all executable tools must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the tool."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A detailed description of what the tool does."""
        pass

    @property
    @abstractmethod
    def parameter_definitions(self) -> Dict[str, Any]:
        """A dictionary defining the parameters expected by the tool."""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool action with arguments."""
        pass

    def get_metadata(self) -> ToolMetadata:
        """Generates the clean ToolMetadata representation of this tool."""
        return ToolMetadata(
            name=self.name,
            description=self.description,
            parameter_definitions=self.parameter_definitions
        )
