import io
import logging
from typing import Any, Dict, Union
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

# Attempt importing PyTorch and Torchvision for EfficientNet
try:
    import torch
    import torchvision.transforms as T
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    torchvision_available = True
except ImportError:
    torchvision_available = False

class CameraDevice(PerceptionDevice):
    """
    Multi-platform camera perception device.
    Uses OpenCV for desktop platforms and classifies the captured frame using EfficientNet.
    """

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager
        self._model = None
        self._weights = None
        self._categories = []
        self._transforms = None
        self._init_efficientnet()

    def _init_efficientnet(self) -> None:
        if not torchvision_available:
            logger.warning("Torchvision not available. Camera classification fallback active.")
            return
        try:
            # Initialize weights and categories
            self._weights = EfficientNet_B0_Weights.DEFAULT
            self._categories = self._weights.meta["categories"]
            self._transforms = self._weights.transforms()
            
            # Load model
            self._model = efficientnet_b0(weights=self._weights)
            self._model.eval()
            logger.info("EfficientNet-B0 loaded successfully for CameraDevice.")
        except Exception as e:
            logger.error(f"Failed to load EfficientNet-B0 for CameraDevice: {e}")
            self._model = None

    @property
    def name(self) -> str:
        return "Camera"

    @property
    def description(self) -> str:
        return "Captures image frames from system camera and classifies them using EfficientNet."

    @property
    def parameter_definitions(self) -> Dict[str, Any]:
        return {
            "resolution": {
                "type": "string",
                "description": "Desired capture resolution (e.g., '1080p', '720p').",
                "required": False
            }
        }

    def capture(self, **kwargs: Any) -> Dict[str, Any]:
        if not self.permission_manager.has_permission("camera"):
            success = self.permission_manager.request_permission("camera")
            if not success:
                raise PermissionError("Camera permission denied.")

        current_os = get_current_os()
        logger.info(f"Initiating camera frame capture on: {current_os}")

        frame = None
        png_bytes = None

        # Desktop execution route
        if current_os in ["windows", "macos", "linux"] and opencv_available:
            try:
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    ret, cv_frame = cap.read()
                    cap.release()
                    if ret:
                        frame = cv_frame
                        is_success, buffer = cv2.imencode(".png", frame)
                        if is_success:
                            png_bytes = buffer.tobytes()
            except Exception as e:
                logger.warning(f"Failed to capture camera frame via OpenCV: {e}")

        # Fallback to simulated transparent black PNG bytes if none captured
        if png_bytes is None:
            png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc` \x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82'

        # Classify the captured frame using EfficientNet
        predictions = []
        if frame is not None and self._model is not None and self._transforms is not None:
            try:
                from PIL import Image
                # Convert BGR frame from OpenCV to PIL RGB Image
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
                
                # Apply model transforms
                input_tensor = self._transforms(pil_image).unsqueeze(0)
                
                # Run prediction
                with torch.no_grad():
                    logits = self._model(input_tensor)
                    probabilities = torch.nn.functional.softmax(logits[0], dim=0)
                    
                # Extract top 3 classes
                top_prob, top_catid = torch.topk(probabilities, 3)
                for i in range(top_prob.size(0)):
                    predictions.append({
                        "class": self._categories[top_catid[i].item()],
                        "confidence": round(float(top_prob[i].item()), 4)
                    })
            except Exception as e:
                logger.warning(f"EfficientNet image classification failed: {e}")

        # If no predictions generated, insert fallback
        if not predictions:
            predictions.append({"class": "unrecognized", "confidence": 1.0})

        return {
            "image_bytes": png_bytes,
            "predictions": predictions
        }
