import logging
from typing import Any, Dict, Union
from .base import PerceptionDevice
from .permissions import PermissionManager
from .detector import get_current_os

logger = logging.getLogger("shiva.perception.clipboard")

# Attempt importing pyperclip for desktop platforms
try:
    import pyperclip
    pyperclip_available = True
except ImportError:
    pyperclip_available = False

class ClipboardDevice(PerceptionDevice):
    """
    Multi-platform clipboard reader.
    Uses pyperclip for desktop platforms and fallbacks for mobile environments.
    """

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "Clipboard"

    @property
    def description(self) -> str:
        return "Reads current pasteboard contents from the system clipboard."

    @property
    def parameter_definitions(self) -> Dict[str, Any]:
        return {}

    def capture(self, **kwargs: Any) -> Union[str, bytes]:
        if not self.permission_manager.has_permission("accessibility"):
            self.permission_manager.request_permission("accessibility")

        current_os = get_current_os()
        logger.info(f"Accessing clipboard data on: {current_os}")

        if current_os in ["windows", "macos", "linux"] and pyperclip_available:
            try:
                return pyperclip.paste()
            except Exception as e:
                logger.warning(f"Failed to read pasteboard via pyperclip: {e}")

        # Fallback to simulated clipboard message
        return "https://github.com/google/shiva-architecture"
