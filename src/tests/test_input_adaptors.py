from pathlib import Path

import pytest

from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import FileInput


def test_file_input_checks_permissions_at_creation(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    input_path = safe_root / "input.txt"
    input_path.write_text("ok", encoding="utf-8")

    permissions = DirectoryPermissions(safe_root)

    adaptor = FileInput(input_path, permissions)

    run_input = adaptor.load()
    assert run_input.text == "ok"
    assert run_input.source_path == str(input_path)


def test_file_input_denies_without_root(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_text("nope", encoding="utf-8")

    permissions = DirectoryPermissions(None)

    with pytest.raises(PermissionError):
        FileInput(input_path, permissions)
