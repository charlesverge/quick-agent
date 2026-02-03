from __future__ import annotations

from pathlib import Path

from quick_agent.directory_permissions import DirectoryPermissions


def read_text(
    path: str,
    permissions: DirectoryPermissions,
) -> str:
    """Read UTF-8 text from a file path."""
    if not permissions.can_read(Path(path)):
        raise PermissionError(f"Read access denied for {path}.")
    safe_path = permissions.resolve(Path(path), for_write=False)
    return safe_path.read_text(encoding="utf-8")
