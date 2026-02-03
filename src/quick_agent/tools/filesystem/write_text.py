from __future__ import annotations

from pathlib import Path

from quick_agent.directory_permissions import DirectoryPermissions


def write_text(
    path: str,
    content: str,
    permissions: DirectoryPermissions,
) -> str:
    """Write UTF-8 text to a file path. Returns the written path."""
    if not permissions.can_write(Path(path)):
        raise PermissionError(f"Write access denied for {path}.")
    out = permissions.resolve(Path(path), for_write=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return str(out)
