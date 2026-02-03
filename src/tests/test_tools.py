from pathlib import Path

from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.tools.filesystem.adapter import FilesystemToolAdapter


def test_write_text_creates_parent_and_writes(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))
    out_path = safe_root / "nested" / "file.txt"
    result = adapter.write_text(str(out_path), "hello")

    assert out_path.read_text(encoding="utf-8") == "hello"
    assert result == str(out_path)


def test_read_text_reads_utf8(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    in_path = safe_root / "in.txt"
    in_path.write_text("data", encoding="utf-8")

    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))
    assert adapter.read_text(str(in_path)) == "data"
