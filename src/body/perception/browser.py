from typing import Any, Dict
from .base import PerceptionDevice
from .permissions import PermissionManager

class BrowserDevice(PerceptionDevice):
    """
    Perception device to collect state metrics of web browsers.
    Does not run browsing steps or execute action events.
    """

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "Browser"

    @property
    def description(self) -> str:
        return "Reads current browser state: current URL, tab count, and DOM snippet if available."

    @property
    def parameter_definitions(self) -> Dict[str, Any]:
        return {}

    def capture(self, **kwargs: Any) -> Dict[str, Any]:
        # Accessibility permission allows reading third-party application UI trees (like browser URL bars)
        if not self.permission_manager.has_permission("accessibility"):
            self.permission_manager.request_permission("accessibility")
        
        # Simulate returning current browser state
        return {
            "current_url": "https://www.google.com",
            "tabs_count": 3,
            "dom_snippet": "<html><body>Google Search Page</body></html>"
        }
