"""Input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path

from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.models.run_input import RunInput


def load_input(path: Path, permissions: DirectoryPermissions) -> RunInput:
    safe_path = permissions.resolve(path, for_write=False)
    if not safe_path.exists():
        raise FileNotFoundError(safe_path)

    if safe_path.suffix.lower() == ".json":
        raw = json.loads(safe_path.read_text(encoding="utf-8"))
        return RunInput(source_path=str(safe_path), kind="json", text=json.dumps(raw, indent=2), data=raw)
    txt = safe_path.read_text(encoding="utf-8")
    return RunInput(source_path=str(safe_path), kind="text", text=txt, data=None)


def ensure_parent_dir(path: Path, permissions: DirectoryPermissions) -> Path:
    safe_path = permissions.resolve(path, for_write=True)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    return safe_path


def write_output(
    path: Path,
    content: str,
    permissions: DirectoryPermissions,
) -> None:
    safe_path = ensure_parent_dir(path, permissions)
    safe_path.write_text(content, encoding="utf-8")
