"""Tool discovery and loading."""

from __future__ import annotations

import importlib
from pathlib import Path
from collections.abc import Callable
from typing import Any

from pydantic_ai.toolsets import FunctionToolset

from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.models.tool_json import ToolJson
from quick_agent.tools.filesystem.adapter import FilesystemToolAdapter
from quick_agent.tools.shell.adapter import ShellToolAdapter


def import_symbol(path: str) -> Any:
    """
    Imports a symbol given "package.module:SymbolName".
    """
    if ":" not in path:
        raise ValueError(f"Expected import path 'module:Symbol', got {path!r}")
    mod, sym = path.split(":", 1)
    module = importlib.import_module(mod)
    return getattr(module, sym)


def _discover_tool_index(tool_roots: list[Path]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for root in tool_roots:
        if not root.exists():
            continue
        for tool_json_path in root.rglob("tool.json"):
            tool_obj = ToolJson.model_validate_json(tool_json_path.read_text(encoding="utf-8"))
            if "." in tool_obj.name:
                raise ValueError(f"Tool name {tool_obj.name!r} must not contain '.'.")
            if tool_obj.name in index:
                continue
            index[tool_obj.name] = tool_json_path
    return index


def load_tools(
    tool_roots: list[Path],
    tool_names: list[str],
    permissions: DirectoryPermissions,
) -> FunctionToolset[Any]:
    """
    Minimal approach: load local python functions and register them into a FunctionToolset.
    """
    toolset = FunctionToolset()

    tool_index = _discover_tool_index(tool_roots)
    fs_adapter = FilesystemToolAdapter(permissions)
    shell_adapter = ShellToolAdapter(permissions)

    for tool_name in tool_names:
        tool_json_path = tool_index.get(tool_name)
        if tool_json_path is None:
            raise FileNotFoundError(f"Missing tool.json for tool {tool_name} in roots: {tool_roots}")

        tool_obj = ToolJson.model_validate_json(tool_json_path.read_text(encoding="utf-8"))
        if tool_obj.impl.kind != "python":
            raise NotImplementedError("Skeleton supports python tools only. Add MCP support next.")

        func: Callable[..., Any]
        if tool_name == "filesystem_read_text":
            func = fs_adapter.read_text
        elif tool_name == "filesystem_write_text":
            func = fs_adapter.write_text
        elif tool_name == "filesystem_append_text":
            func = fs_adapter.append_text
        elif tool_name == "filesystem_list_files":
            func = fs_adapter.list_files
        elif tool_name == "filesystem_delete_file":
            func = fs_adapter.delete_file
        elif tool_name == "filesystem_find_closest_file":
            func = fs_adapter.find_closest_file
        elif tool_name == "shell_run":
            func = shell_adapter.run
        else:
            func = import_symbol(f"{tool_obj.impl.module}:{tool_obj.impl.function}")

        toolset.add_function(func=func, name=tool_obj.name, description=tool_obj.description)

    return toolset
