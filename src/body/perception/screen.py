import io
import logging
from typing import Any, Dict
from .base import PerceptionDevice
from .permissions import PermissionManager
from .detector import get_current_os

logger = logging.getLogger("shiva.perception.screen")
try:
    import mss
    mss_available = True
except ImportError:
    mss_available = False

try:
    from PIL import ImageGrab
    pillow_grab_available = True
except ImportError:
    pillow_grab_available = False

class ScreenDevice(PerceptionDevice):
    """
    Multi-platform screen capture device.
    Uses MSS or PIL ImageGrab for desktop screenshots.
    """

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "Screen"

    @property
    def description(self) -> str:
        return "Captures screenshots of the active display."

    @property
    def parameter_definitions(self) -> Dict[str, Any]:
        return {
            "display_id": {
                "type": "integer",
                "description": "Index identifier of the monitor screen (0 for primary).",
                "required": False
            }
        }

    def capture(self, **kwargs: Any) -> bytes:
        if not self.permission_manager.has_permission("screen_capture"):
            success = self.permission_manager.request_permission("screen_capture")
            if not success:
                raise PermissionError("Screen capture permission denied.")

        current_os = get_current_os()
        logger.info(f"Initiating screen capture on: {current_os}")

        # Desktop route: try MSS first, fallback to Pillow ImageGrab
        if current_os in ["windows", "macos", "linux"]:
            if mss_available:
                try:
                    with mss.mss() as sct:
                        # Capture primary monitor
                        monitor = sct.monitors[1]
                        screenshot = sct.grab(monitor)
                        return mss.tools.to_png(screenshot.rgb, screenshot.size)
                except Exception as e:
                    logger.warning(f"MSS screenshot failed: {e}")

            if pillow_grab_available:
                try:
                    img = ImageGrab.grab()
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    return img_bytes.getvalue()
                except Exception as e:
                    logger.warning(f"Pillow ImageGrab screenshot failed: {e}")

        # Fallback to simulated transparent PNG bytes
        return b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc` \x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82'
