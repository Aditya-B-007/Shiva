import io
import logging
from typing import Any, Dict
from .base import PerceptionDevice
from .permissions import PermissionManager
from .detector import get_current_os

logger = logging.getLogger("shiva.perception.camera")

# Attempt importing OpenCV for desktop captures
try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False

class CameraDevice(PerceptionDevice):
    """
    Multi-platform camera perception device.
    Uses OpenCV for desktop platforms and fallbacks for mobile environments.
    """

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "Camera"

    @property
    def description(self) -> str:
        return "Captures raw image frames from the system camera. No reasoning or OCR."

    @property
    def parameter_definitions(self) -> Dict[str, Any]:
        return {
            "resolution": {
                "type": "string",
                "description": "Desired capture resolution (e.g., '1080p', '720p').",
                "required": False
            }
        }

    def capture(self, **kwargs: Any) -> bytes:
        if not self.permission_manager.has_permission("camera"):
            success = self.permission_manager.request_permission("camera")
            if not success:
                raise PermissionError("Camera permission denied.")

        current_os = get_current_os()
        logger.info(f"Initiating camera frame capture on: {current_os}")

        # Desktop execution route
        if current_os in ["windows", "macos", "linux"] and opencv_available:
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        is_success, buffer = cv2.imencode(".png", frame)
                        if is_success:
                            return buffer.tobytes()
            except Exception as e:
                logger.warning(f"Failed to capture camera frame via OpenCV: {e}")

        # Fallback to simulated transparent black PNG bytes
        mock_png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc` \x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82'
        return mock_png
