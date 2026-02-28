"""Filesystem tool adapter with directory permissions."""

from __future__ import annotations

import difflib
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

    def append_text(self, path: str, content: str) -> str:
        if not self._permissions.can_write(Path(path)):
            raise PermissionError(f"Write access denied for {path}.")
        safe_path = self._permissions.resolve(Path(path), for_write=True)
        safe_path.parent.mkdir(parents=True, exist_ok=True)
        with safe_path.open("a", encoding="utf-8") as fh:
            fh.write(content)
        return str(safe_path)

    def list_files(self, directory: str) -> str:
        if not self._permissions.can_read(Path(directory)):
            raise PermissionError(f"Read access denied for {directory}.")
        resolved = self._permissions.resolve(Path(directory), for_write=False)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        entries: list[str] = []
        for entry in sorted(resolved.iterdir()):
            name = entry.name + "/" if entry.is_dir() else entry.name
            entries.append(name)
        return "\n".join(entries)

    def delete_file(self, path: str) -> str:
        if not self._permissions.can_write(Path(path)):
            raise PermissionError(f"Write access denied for {path}.")
        resolved = self._permissions.resolve(Path(path), for_write=True)
        if not resolved.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if resolved.is_dir():
            raise IsADirectoryError(f"Path is a directory, not a file: {path}")
        resolved.unlink()
        return f"Deleted: {resolved}"

    def find_closest_file(self, directory: str, name: str) -> str:
        if not self._permissions.can_read(Path(directory)):
            raise PermissionError(f"Read access denied for {directory}.")
        resolved = self._permissions.resolve(Path(directory), for_write=False)
        if not resolved.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        file_names = [entry.name for entry in resolved.iterdir() if entry.is_file()]
        if not file_names:
            return ""
        matches = difflib.get_close_matches(name, file_names, n=1, cutoff=0.0)
        if not matches:
            return ""
        return str(resolved / matches[0])
