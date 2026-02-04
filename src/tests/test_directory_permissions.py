from pathlib import Path

import pytest

from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.orchestrator import Orchestrator
from quick_agent.quick_agent import QuickAgent


class AsyncReturner:
    def __init__(self, value: object) -> None:
        self.value = value

    async def __call__(self, *args: object, **kwargs: object) -> object:
        return self.value


def test_directory_permissions_resolve_allows_within_root(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    perms = DirectoryPermissions(safe_root)

    resolved = perms.resolve(Path("nested/file.txt"), for_write=False)

    assert resolved == safe_root / "nested" / "file.txt"


def test_directory_permissions_resolve_blocks_escape(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    perms = DirectoryPermissions(safe_root)

    with pytest.raises(PermissionError):
        perms.resolve(Path("../outside.txt"), for_write=False)


def test_directory_permissions_can_read_write(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    perms = DirectoryPermissions(safe_root)

    assert perms.can_read(Path("ok.txt")) is True
    assert perms.can_write(Path("ok.txt")) is True
    assert perms.can_read(Path("../nope.txt")) is False
    assert perms.can_write(Path("../nope.txt")) is False


def test_directory_permissions_without_root_denies_all() -> None:
    perms = DirectoryPermissions(None)

    with pytest.raises(PermissionError):
        perms.resolve(Path("anything.txt"), for_write=False)

    assert perms.can_read(Path("anything.txt")) is False
    assert perms.can_write(Path("anything.txt")) is False


def test_agent_cannot_write_outside_scoped_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    agent_md = """---
name: "Scoped Agent"
safe_dir: "agent"
chain:
  - id: one
    kind: text
    prompt_section: step:one
output:
  format: json
  file: ../out.json
---

## step:one

Say ok.
"""
    (agents_dir / "scoped.md").write_text(agent_md, encoding="utf-8")

    input_path = safe_root / "agent" / "input.txt"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text("hi", encoding="utf-8")

    orch = Orchestrator(
        [agents_dir],
        [tmp_path / "tools"],
        safe_dir=safe_root,
    )
    monkeypatch.setattr(QuickAgent, "_run_chain", AsyncReturner("ok"))

    with pytest.raises(PermissionError):
        import anyio

        anyio.run(orch.run, "scoped", input_path)
