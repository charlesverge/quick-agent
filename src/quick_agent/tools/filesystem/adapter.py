"""Filesystem tool adapter with directory permissions."""

from __future__ import annotations

from pathlib import Path

from quick_agent.directory_permissions import DirectoryPermissions


class FilesystemToolAdapter:
    def __init__(self, permissions: DirectoryPermissions) -> None:
        self._permissions = permissions

    def read_text(self, path: str) -> str:
        if not self._permissions.can_read(Path(path)):
            raise PermissionError(f"Read access denied for {path}.")
        safe_path = self._permissions.resolve(Path(path), for_write=False)
        return safe_path.read_text(encoding="utf-8")

    def write_text(self, path: str, content: str) -> str:
        if not self._permissions.can_write(Path(path)):
            raise PermissionError(f"Write access denied for {path}.")
        safe_path = self._permissions.resolve(Path(path), for_write=True)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        safe_path.write_text(content, encoding="utf-8")
        return str(safe_path)
