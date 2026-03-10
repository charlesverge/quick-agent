from __future__ import annotations

import subprocess
from pathlib import Path
import shlex

from quick_agent.directory_permissions import DirectoryPermissions


class ShellToolAdapter:
    def __init__(self, permissions: DirectoryPermissions) -> None:
        self._permissions = permissions

    def _resolve_cwd(self, cwd: str | None) -> Path:
        if cwd is None or cwd.strip() == "":
            root = self._permissions.root
            if root is None:
                raise PermissionError("No safe directory configured for shell_run.")
            return root
        cwd_path = Path(cwd)
        if not self._permissions.can_read(cwd_path):
            raise PermissionError(f"Read access denied for cwd {cwd}.")
        resolved = self._permissions.resolve(cwd_path, for_write=False)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {cwd}")
        return resolved

    def run(self, command: str, cwd: str | None = None, timeout_seconds: int = 30) -> str:
        trimmed = command.strip()
        if trimmed == "":
            raise ValueError("Command must not be empty.")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0.")

        resolved_cwd = self._resolve_cwd(cwd)
        args = shlex.split(trimmed)
        completed = subprocess.run(
            args,
            cwd=str(resolved_cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()
        return (
            f"returncode: {completed.returncode}\n"
            f"cwd: {resolved_cwd}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
