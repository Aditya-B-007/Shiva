import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

class WorkspaceContext:
    """
    Manages access and operations (hooks) within a specific target workspace directory.
    Provides utility methods to read/edit files and execute commands safely.
    """
    def __init__(self, root_path: str) -> None:
        self.root_path = Path(root_path).resolve()
        if not self.root_path.exists():
            raise FileNotFoundError(f"Target workspace directory does not exist: {self.root_path}")

    def _resolve_path(self, target_path: str) -> Path:
        """Resolves target paths to be absolute and verifies they are within the workspace root."""
        resolved = Path(target_path)
        if not resolved.is_absolute():
            resolved = (self.root_path / resolved).resolve()
        else:
            resolved = resolved.resolve()
        
        # Enforce sandbox: check if target is inside root_path
        try:
            resolved.relative_to(self.root_path)
        except ValueError:
            raise PermissionError(f"Access denied: path {resolved} is outside the workspace root {self.root_path}")
        return resolved

    def list_dir(self, sub_path: str = "") -> List[Dict[str, Any]]:
        """Lists files and folders in the workspace at the given sub_path."""
        target = self._resolve_path(sub_path)
        if not target.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {target}")
            
        results = []
        for entry in os.scandir(target):
            stat = entry.stat()
            results.append({
                "name": entry.name,
                "is_dir": entry.is_dir(),
                "size_bytes": stat.st_size if entry.is_file() else 0,
                "relative_path": str(Path(entry.path).relative_to(self.root_path))
            })
        return results

    def view_file(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """Reads content of a file from start_line to end_line (1-indexed, inclusive)."""
        target = self._resolve_path(file_path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
            
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            
        if start_line is None:
            start_line = 1
        if end_line is None:
            end_line = len(lines)
            
        # Clamp bounds
        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)
        
        return "".join(lines[start_idx:end_idx])

    def write_to_file(self, file_path: str, content: str) -> None:
        """Writes content to a file, creating parent directories if they don't exist."""
        target = self._resolve_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(content)

    def replace_file_content(self, file_path: str, target_content: str, replacement_content: str) -> None:
        """Replaces exact target content with replacement content in the file."""
        target = self._resolve_path(file_path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
            
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
            
        if target_content not in content:
            raise ValueError("Target content to replace not found in the file.")
            
        new_content = content.replace(target_content, replacement_content)
        with open(target, "w", encoding="utf-8") as f:
            f.write(new_content)

    def grep_search(self, query: str) -> List[Dict[str, Any]]:
        """Performs simple search matching inside text files in the workspace."""
        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        
        # Fallback to python walk since we want robust on-device running
        for root, _, files in os.walk(self.root_path):
            # Skip common VCS/compiled/temp folders
            if any(p in root for p in [".git", "__pycache__", "build", "dist", ".gemini", ".agents"]):
                continue
            for file in files:
                full_path = Path(root) / file
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line_num, line in enumerate(f, 1):
                            if pattern.search(line):
                                results.append({
                                    "file": str(full_path.relative_to(self.root_path)),
                                    "line_number": line_num,
                                    "content": line.strip()
                                })
                except Exception:
                    pass
        return results[:100]  # Cap results

    def run_command(self, command: str) -> Dict[str, Any]:
        """Executes a terminal shell command inside the workspace directory."""
        try:
            # Use shell=True for windows command execution compatibility
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.root_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60
            )
            return {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr
            }
        except subprocess.TimeoutExpired as e:
            return {
                "exit_code": -1,
                "stdout": e.stdout or "",
                "stderr": "Command timed out after 60 seconds."
            }
        except Exception as e:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Failed to execute command: {str(e)}"
            }
