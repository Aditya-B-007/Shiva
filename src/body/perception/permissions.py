import logging
from typing import Dict
from .detector import get_current_os

logger = logging.getLogger("shiva.perception.permissions")
try:
    from jnius import autoclass
    android_available = True
except ImportError:
    android_available = False

try:
    from rubicon.objc import ObjCClass
    ios_available = True
except ImportError:
    ios_available = False

class PermissionManager:
    """
    Handles permission queries and dynamic permission requesting across multiple OS platforms.
    """

    def __init__(self) -> None:
        self._mock_permissions: Dict[str, bool] = {
            "camera": False,
            "microphone": False,
            "accessibility": False,
            "screen_capture": False,
            "notifications": False
        }

    def has_permission(self, permission_name: str) -> bool:
        name = permission_name.lower().strip()
        current_os = get_current_os()
        logger.info(f"Checking permission '{name}' on platform: {current_os}")

        if current_os == "android" and android_available:
            try:
                ContextCompat = autoclass("androidx.core.content.ContextCompat")
                Activity = autoclass("android.app.Activity")
                # Map to standard Android Manifest Permission Strings
                perm_map = {
                    "camera": "android.permission.CAMERA",
                    "microphone": "android.permission.RECORD_AUDIO",
                }
                android_perm = perm_map.get(name)
                if not android_perm:
                    return True # Default to True for unmapped/implicit permissions
                
                # Check permission status via ContextCompat
                # Note: Assumes Python Activity context is obtainable or mapped
                context = autoclass("org.kivy.android.PythonActivity").mActivity
                PackageManager = autoclass("android.content.pm.PackageManager")
                result = ContextCompat.checkSelfPermission(context, android_perm)
                return result == PackageManager.PERMISSION_GRANTED
            except Exception as e:
                logger.error(f"Android permission check failed: {e}")
                return False

        elif current_os == "ios" and ios_available:
            try:
                # Check AVFoundation camera/microphone permissions using Rubicon-ObjC
                AVMediaTypeVideo = "vide"
                AVMediaTypeAudio = "soun"
                AVCaptureDevice = ObjCClass("AVCaptureDevice")
                
                media_type = AVMediaTypeVideo if name == "camera" else AVMediaTypeAudio
                status = AVCaptureDevice.authorizationStatusForMediaType(media_type)
                # 3 corresponds to AVAuthorizationStatusAuthorized
                return status == 3
            except Exception as e:
                logger.error(f"iOS permission check failed: {e}")
                return False

        # Desktops (macOS, Windows, Linux) - fallback mock or assuming system permission prompts
        return self._mock_permissions.get(name, True)

    def request_permission(self, permission_name: str) -> bool:
        name = permission_name.lower().strip()
        current_os = get_current_os()
        logger.info(f"Requesting permission '{name}' on platform: {current_os}")

        if current_os == "android" and android_available:
            try:
                ActivityCompat = autoclass("androidx.core.app.ActivityCompat")
                perm_map = {
                    "camera": "android.permission.CAMERA",
                    "microphone": "android.permission.RECORD_AUDIO",
                }
                android_perm = perm_map.get(name)
                if not android_perm:
                    return True
                
                activity = autoclass("org.kivy.android.PythonActivity").mActivity
                # Request permissions from Android framework context
                ActivityCompat.requestPermissions(activity, [android_perm], 101)
                return True
            except Exception as e:
                logger.error(f"Android permission request failed: {e}")
                return False

        elif current_os == "ios" and ios_available:
            try:
                AVMediaTypeVideo = "vide"
                AVMediaTypeAudio = "soun"
                AVCaptureDevice = ObjCClass("AVCaptureDevice")
                
                media_type = AVMediaTypeVideo if name == "camera" else AVMediaTypeAudio
                # Requests dynamic access permission using native AVCaptureDevice block handlers
                AVCaptureDevice.requestAccessForMediaType(media_type, completionHandler=None)
                return True
            except Exception as e:
                logger.error(f"iOS permission request failed: {e}")
                return False
        self._mock_permissions[name] = True
        return True
