import os
from typing import Any, Dict
from .base import PerceptionDevice
from .permissions import PermissionManager

class FilesystemDevice(PerceptionDevice):

    def __init__(self, permission_manager: PermissionManager) -> None:
        self.permission_manager = permission_manager

    @property
    def name(self) -> str:
        return "Filesystem"

    @property
    def description(self) -> str:
        return "Reads file path metadata or directory entries. Does not interpret code."

    @property
    def parameter_definitions(self) -> Dict[str, Any]:
        return {
            "path": {
                "type": "string",
                "description": "Absolute file or directory path.",
                "required": True
            }
        }

    def capture(self, **kwargs: Any) -> Dict[str, Any]:
        path = kwargs.get("path")
        if not path:
            raise ValueError("Parameter 'path' is required.")

        # Simulate returning basic metadata of the path
        exists = os.path.exists(path)
        is_dir = os.path.isdir(path) if exists else False
        size = os.path.getsize(path) if exists and not is_dir else 0
        
        return {
            "path": path,
            "exists": exists,
            "is_directory": is_dir,
            "size_bytes": size,
            "timestamp": os.path.getmtime(path) if exists else 0.0
        }
