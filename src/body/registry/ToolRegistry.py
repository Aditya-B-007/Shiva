from typing import Dict, List, Optional
from ..tools.base import Tool
from ..models.ToolMetadata import ToolMetadata

class ToolRegistry:
    """Registry that holds tool instances and provides metadata query support."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Registers a tool inside the execution system registry."""
        if not tool.name:
            raise ValueError("Tool name cannot be empty")
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """Resolves and returns a registered tool instance by name."""
        return self._tools.get(name)

    def get_all_metadata(self) -> List[ToolMetadata]:
        """Exposes metadata definitions for all registered tools to the Decoder."""
        return [tool.get_metadata() for tool in self._tools.values()]
