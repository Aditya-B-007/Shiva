import time
from typing import Any, Dict
from .base import Tool
from ..models.ToolResult import ToolResult

class LaunchApplication(Tool):
    """Tool that launches a desktop application by name."""

    @property
    def name(self) -> str:
        return "LaunchApplication"

    @property
    def description(self) -> str:
        return "Launches a desktop application by name."

    @property
    def parameter_definitions(self) -> Dict[str, Any]:
        return {
            "application": {
                "type": "string",
                "description": "The name of the application to launch.",
                "required": True
            }
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        app_name = kwargs.get("application")
        if not app_name:
            return ToolResult(
                success=False,
                output=None,
                error="Parameter 'application' is missing or empty.",
                timestamp=time.time()
            )
        print(f"[Device OS] Simulated launching app: {app_name}")
        return ToolResult(
            success=True,
            output=f"Successfully launched {app_name}",
            timestamp=time.time()
        )

class MouseClick(Tool):
    """Tool that clicks at specific screen coordinates."""

    @property
    def name(self) -> str:
        return "MouseClick"

    @property
    def description(self) -> str:
        return "Clicks at a screen coordinate."

    @property
    def parameter_definitions(self) -> Dict[str, Any]:
        return {
            "x": {
                "type": "integer",
                "description": "X coordinate on screen.",
                "required": True
            },
            "y": {
                "type": "integer",
                "description": "Y coordinate on screen.",
                "required": True
            }
        }

    def execute(self, **kwargs: Any) -> ToolResult:
        x = kwargs.get("x")
        y = kwargs.get("y")
        if x is None or y is None:
            return ToolResult(
                success=False,
                output=None,
                error="Parameters 'x' and 'y' are required.",
                timestamp=time.time()
            )
        print(f"[Device OS] Simulated mouse click at x={x}, y={y}")
        return ToolResult(
            success=True,
            output=f"Successfully clicked screen coordinate ({x}, {y})",
            timestamp=time.time()
        )
