from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.tools import Tool
from pydantic_ai.toolsets import FunctionToolset
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.tools.filesystem.adapter import FilesystemToolAdapter
from quick_agent.tools.shell.adapter import ShellToolAdapter
from quick_agent.tools_loader import (
    _discover_tool_index,
    load_tool_definitions,
    load_tools,
)


def _system_tools_dir() -> Path:
    import quick_agent.tools as _tools_pkg

    return Path(_tools_pkg.__file__).resolve().parent


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


# --- append_text ---


def test_append_text_creates_file_on_first_call(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))
    out_path = safe_root / "log.txt"

    result = adapter.append_text(str(out_path), "line1\n")

    assert out_path.read_text(encoding="utf-8") == "line1\n"
    assert result == str(out_path)


def test_append_text_accumulates_content(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))
    out_path = safe_root / "log.txt"

    adapter.append_text(str(out_path), "line1\n")
    adapter.append_text(str(out_path), "line2\n")

    assert out_path.read_text(encoding="utf-8") == "line1\nline2\n"


def test_append_text_creates_parent_directories(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))
    out_path = safe_root / "nested" / "deep" / "log.txt"

    adapter.append_text(str(out_path), "data")

    assert out_path.read_text(encoding="utf-8") == "data"


def test_append_text_denies_outside_root(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.txt"
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    with pytest.raises(PermissionError):
        adapter.append_text(str(outside), "data")


# --- list_files ---


def test_list_files_returns_sorted_entries(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    (safe_root / "alpha.txt").write_text("a", encoding="utf-8")
    (safe_root / "beta.txt").write_text("b", encoding="utf-8")
    sub_dir = safe_root / "sub"
    sub_dir.mkdir()
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    result = adapter.list_files(str(safe_root))

    entries = result.split("\n")
    assert "alpha.txt" in entries
    assert "beta.txt" in entries
    assert "sub/" in entries
    assert entries == sorted(entries)


def test_list_files_marks_directories_with_slash(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    (safe_root / "mydir").mkdir()
    (safe_root / "myfile.txt").write_text("", encoding="utf-8")
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    result = adapter.list_files(str(safe_root))

    assert "mydir/" in result
    assert "myfile.txt" in result
    assert "myfile.txt/" not in result


def test_list_files_returns_empty_string_for_empty_directory(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    result = adapter.list_files(str(safe_root))

    assert result == ""


def test_list_files_denies_outside_root(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    with pytest.raises(PermissionError):
        adapter.list_files(str(outside))


def test_list_files_raises_for_non_directory(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    target = safe_root / "file.txt"
    target.write_text("content", encoding="utf-8")
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    with pytest.raises(NotADirectoryError):
        adapter.list_files(str(target))


# --- delete_file ---


def test_delete_file_removes_file(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    target = safe_root / "to_delete.txt"
    target.write_text("gone", encoding="utf-8")
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    result = adapter.delete_file(str(target))

    assert not target.exists()
    assert "Deleted" in result


def test_delete_file_raises_if_missing(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    with pytest.raises(FileNotFoundError):
        adapter.delete_file(str(safe_root / "nonexistent.txt"))


def test_delete_file_raises_for_directory(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    subdir = safe_root / "subdir"
    subdir.mkdir()
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    with pytest.raises(IsADirectoryError):
        adapter.delete_file(str(subdir))


def test_delete_file_denies_outside_root(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "target.txt"
    outside.write_text("sensitive", encoding="utf-8")
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    with pytest.raises(PermissionError):
        adapter.delete_file(str(outside))


# --- find_closest_file ---


def test_find_closest_file_returns_best_match(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    (safe_root / "meeting_notes.txt").write_text("", encoding="utf-8")
    (safe_root / "report_final.txt").write_text("", encoding="utf-8")
    (safe_root / "readme.md").write_text("", encoding="utf-8")
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    result = adapter.find_closest_file(str(safe_root), "meeting")

    assert result.endswith("meeting_notes.txt")


def test_find_closest_file_returns_full_path(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    notes_path = safe_root / "notes.txt"
    notes_path.write_text("", encoding="utf-8")
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    result = adapter.find_closest_file(str(safe_root), "notes")

    assert result == str(notes_path)


def test_find_closest_file_returns_empty_for_empty_directory(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    result = adapter.find_closest_file(str(safe_root), "anything")

    assert result == ""


def test_find_closest_file_ignores_subdirectories(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    (safe_root / "notes").mkdir()
    (safe_root / "notes.txt").write_text("", encoding="utf-8")
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    result = adapter.find_closest_file(str(safe_root), "notes")

    assert result.endswith("notes.txt")
    assert not result.endswith("/")


def test_find_closest_file_denies_outside_root(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    with pytest.raises(PermissionError):
        adapter.find_closest_file(str(outside), "file")


# --- shell run ---


def test_shell_run_executes_command_in_safe_root(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    adapter = ShellToolAdapter(DirectoryPermissions(safe_root))

    result = adapter.run("pwd")

    assert "returncode: 0" in result
    assert f"cwd: {safe_root}" in result
    assert str(safe_root) in result


def test_shell_run_denies_cwd_outside_root(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir(parents=True, exist_ok=True)
    adapter = ShellToolAdapter(DirectoryPermissions(safe_root))

    with pytest.raises(PermissionError):
        adapter.run("pwd", cwd=str(outside))


# --- load_tools dispatch cycle ---


class _CapturingToolset(FunctionToolset[Any]):
    """FunctionToolset subclass that records what is registered."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[Any, str]] = []

    def add_function(self, *args: Any, **kwargs: Any) -> Tool[Any]:
        func = args[0] if args else kwargs.get("func")
        name = kwargs.get("name")
        if func is not None and name is not None:
            self.calls.append((func, name))
        return super().add_function(*args, **kwargs)


def test_load_tools_dispatches_append_text_to_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    permissions = DirectoryPermissions(safe_root)
    capturing = _CapturingToolset()
    import quick_agent.tools_loader as tl_module

    monkeypatch.setattr(tl_module, "FunctionToolset", lambda: capturing)

    load_tools([_system_tools_dir()], ["filesystem_append_text"], permissions)

    assert len(capturing.calls) == 1
    func, name = capturing.calls[0]
    assert name == "filesystem_append_text"
    assert isinstance(func.__self__, FilesystemToolAdapter)
    assert func.__name__ == "append_text"


def test_load_tools_dispatches_list_files_to_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    permissions = DirectoryPermissions(safe_root)
    capturing = _CapturingToolset()
    import quick_agent.tools_loader as tl_module

    monkeypatch.setattr(tl_module, "FunctionToolset", lambda: capturing)

    load_tools([_system_tools_dir()], ["filesystem_list_files"], permissions)

    assert len(capturing.calls) == 1
    func, name = capturing.calls[0]
    assert name == "filesystem_list_files"
    assert isinstance(func.__self__, FilesystemToolAdapter)
    assert func.__name__ == "list_files"


def test_load_tools_dispatches_delete_file_to_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    permissions = DirectoryPermissions(safe_root)
    capturing = _CapturingToolset()
    import quick_agent.tools_loader as tl_module

    monkeypatch.setattr(tl_module, "FunctionToolset", lambda: capturing)

    load_tools([_system_tools_dir()], ["filesystem_delete_file"], permissions)

    assert len(capturing.calls) == 1
    func, name = capturing.calls[0]
    assert name == "filesystem_delete_file"
    assert isinstance(func.__self__, FilesystemToolAdapter)
    assert func.__name__ == "delete_file"


def test_load_tools_dispatches_find_closest_file_to_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    permissions = DirectoryPermissions(safe_root)
    capturing = _CapturingToolset()
    import quick_agent.tools_loader as tl_module

    monkeypatch.setattr(tl_module, "FunctionToolset", lambda: capturing)

    load_tools([_system_tools_dir()], ["filesystem_find_closest_file"], permissions)

    assert len(capturing.calls) == 1
    func, name = capturing.calls[0]
    assert name == "filesystem_find_closest_file"
    assert isinstance(func.__self__, FilesystemToolAdapter)
    assert func.__name__ == "find_closest_file"


def test_load_tools_dispatches_shell_run_to_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    permissions = DirectoryPermissions(safe_root)
    capturing = _CapturingToolset()
    import quick_agent.tools_loader as tl_module

    monkeypatch.setattr(tl_module, "FunctionToolset", lambda: capturing)

    load_tools([_system_tools_dir()], ["shell_run"], permissions)

    assert len(capturing.calls) == 1
    func, name = capturing.calls[0]
    assert name == "shell_run"
    assert isinstance(func.__self__, ShellToolAdapter)
    assert func.__name__ == "run"


# --- multi-step operation cycle ---


def test_list_find_read_append_full_cycle(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    (safe_root / "meeting_notes.txt").write_text(
        "Original content.\n", encoding="utf-8"
    )
    (safe_root / "report_q4.txt").write_text("Q4 data.\n", encoding="utf-8")
    (safe_root / "readme.md").write_text("# README\n", encoding="utf-8")

    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    # Step 1: list
    listing = adapter.list_files(str(safe_root))
    entries = listing.split("\n")
    assert "meeting_notes.txt" in entries
    assert "report_q4.txt" in entries
    assert "readme.md" in entries

    # Step 2: find closest
    found_path = adapter.find_closest_file(str(safe_root), "meeting")
    assert found_path.endswith("meeting_notes.txt")

    # Step 3: read
    content = adapter.read_text(found_path)
    assert content == "Original content.\n"

    # Step 4: append
    adapter.append_text(found_path, "Appended line.\n")

    # Step 5: verify on disk
    final = (safe_root / "meeting_notes.txt").read_text(encoding="utf-8")
    assert final == "Original content.\nAppended line.\n"


def test_write_find_delete_cycle(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    safe_root.mkdir(parents=True, exist_ok=True)
    adapter = FilesystemToolAdapter(DirectoryPermissions(safe_root))

    # Step 1: write two files
    adapter.write_text(str(safe_root / "temp_alpha.txt"), "alpha")
    adapter.write_text(str(safe_root / "temp_beta.txt"), "beta")

    # Step 2: find the 'alpha' file
    found_path = adapter.find_closest_file(str(safe_root), "temp_alpha")
    assert found_path.endswith("temp_alpha.txt")

    # Step 3: delete it
    result = adapter.delete_file(found_path)
    assert "Deleted" in result
    assert not (safe_root / "temp_alpha.txt").exists()

    # Step 4: list to confirm only beta remains
    listing = adapter.list_files(str(safe_root))
    assert "temp_beta.txt" in listing
    assert "temp_alpha.txt" not in listing


# --- load_tool_definitions schema introspection ---


def test_load_tool_definitions_generates_schema_for_builtin_tool() -> None:
    tools_root = _system_tools_dir()
    defs = load_tool_definitions([tools_root], ["filesystem_list_files"])

    assert len(defs) == 1
    assert defs[0].name == "filesystem_list_files"
    props = defs[0].input_schema.get("properties")
    assert isinstance(props, dict)
    assert "directory" in props


def test_load_tool_definitions_generates_schema_for_multi_param_tool() -> None:
    tools_root = _system_tools_dir()
    defs = load_tool_definitions([tools_root], ["filesystem_write_text"])

    assert len(defs) == 1
    props = defs[0].input_schema.get("properties")
    assert isinstance(props, dict)
    assert "path" in props
    assert "content" in props
    required = defs[0].input_schema.get("required")
    assert isinstance(required, list)
    assert "path" in required
    assert "content" in required


def test_load_tool_definitions_generates_schema_for_shell_run() -> None:
    tools_root = _system_tools_dir()
    defs = load_tool_definitions([tools_root], ["shell_run"])

    assert len(defs) == 1
    props = defs[0].input_schema.get("properties")
    assert isinstance(props, dict)
    assert "command" in props
    required = defs[0].input_schema.get("required")
    assert isinstance(required, list)
    assert "command" in required


def test_load_tool_definitions_multiple_tools() -> None:
    tools_root = _system_tools_dir()
    names = ["filesystem_list_files", "filesystem_read_text", "shell_run"]
    defs = load_tool_definitions([tools_root], names)

    assert len(defs) == 3
    result_names = [d.name for d in defs]
    assert result_names == names
    for d in defs:
        assert isinstance(d.input_schema.get("properties"), dict)


def test_load_tool_definitions_raises_for_missing_tool() -> None:
    tools_root = _system_tools_dir()
    with pytest.raises(FileNotFoundError, match="nonexistent_tool"):
        load_tool_definitions([tools_root], ["nonexistent_tool"])


def test_load_tool_definitions_schema_matches_interactive_mode() -> None:
    tools_root = _system_tools_dir()
    permissions = DirectoryPermissions(Path("/tmp"))
    toolset = load_tools([tools_root], ["filesystem_list_files"], permissions)
    interactive_schema = toolset.tools[
        "filesystem_list_files"
    ].function_schema.json_schema

    defs = load_tool_definitions([tools_root], ["filesystem_list_files"])
    batch_schema = defs[0].input_schema

    assert batch_schema == interactive_schema


# --- _discover_tool_index validation ---


def _write_tool_json(root: Path, name: str, content: str) -> None:
    tool_dir = root / name
    tool_dir.mkdir(parents=True, exist_ok=True)
    (tool_dir / "tool.json").write_text(content, encoding="utf-8")


def test_discover_tool_index_rejects_empty_module(tmp_path: Path) -> None:
    _write_tool_json(
        tmp_path,
        "bad_tool",
        '{"name": "bad_tool", "impl": {"kind": "python", "module": "", "function": "run"}}',
    )
    with pytest.raises(ValueError, match="missing required impl fields"):
        _discover_tool_index([tmp_path])


def test_discover_tool_index_rejects_empty_function(tmp_path: Path) -> None:
    _write_tool_json(
        tmp_path,
        "bad_tool",
        '{"name": "bad_tool", "impl": {"kind": "python", "module": "some.module", "function": ""}}',
    )
    with pytest.raises(ValueError, match="missing required impl fields"):
        _discover_tool_index([tmp_path])


def test_discover_tool_index_rejects_unsupported_kind(tmp_path: Path) -> None:
    _write_tool_json(
        tmp_path,
        "bad_tool",
        '{"name": "bad_tool", "impl": {"kind": "mcp", "module": "x", "function": "y"}}',
    )
    with pytest.raises(ValueError, match="unsupported impl.kind"):
        _discover_tool_index([tmp_path])


def test_discover_tool_index_accepts_valid_tool(tmp_path: Path) -> None:
    _write_tool_json(
        tmp_path,
        "ok_tool",
        '{"name": "ok_tool", "impl": {"kind": "python", "module": "some.mod", "function": "run"}}',
    )
    index = _discover_tool_index([tmp_path])
    assert "ok_tool" in index
    assert index["ok_tool"].impl.module == "some.mod"
    assert index["ok_tool"].impl.function == "run"
