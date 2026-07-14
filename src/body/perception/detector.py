import os
import sys
import platform

def get_current_os() -> str:
    # 1. Check for Android environment indicators
    # Standard Python platform module may return 'linux' on Android.
    if sys.platform == "android" or "ANDROID_BOOTLOGO" in os.environ or "ANDROID_ROOT" in os.environ:
        return "android"
        
    # 2. Check for iOS indicators
    if sys.platform == "ios" or (platform.system() == "Darwin" and platform.machine().startswith("iP")):
        return "ios"
        
    # 3. Check standard OS platforms
    sys_name = platform.system().lower()
    if "darwin" in sys_name:
        return "macos"
    elif "windows" in sys_name:
        return "windows"
    elif "linux" in sys_name:
        return "linux"
        
    return "unknown"
