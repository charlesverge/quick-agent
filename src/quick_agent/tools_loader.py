"""Tool discovery and loading."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from pydantic_ai.toolsets import FunctionToolset

from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.models.tool_json import ToolJson
from quick_agent.tools.filesystem.adapter import FilesystemToolAdapter


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
            if tool_obj.id in index:
                continue
            index[tool_obj.id] = tool_json_path
    return index


def load_tools(
    tool_roots: list[Path],
    tool_ids: list[str],
    permissions: DirectoryPermissions,
) -> FunctionToolset[Any]:
    """
    Minimal approach: load local python functions and register them into a FunctionToolset.
    """
    toolset = FunctionToolset()

    tool_index = _discover_tool_index(tool_roots)
    fs_adapter = FilesystemToolAdapter(permissions)

    for tool_id in tool_ids:
        tool_json_path = tool_index.get(tool_id)
        if tool_json_path is None:
            raise FileNotFoundError(f"Missing tool.json for tool {tool_id} in roots: {tool_roots}")

        tool_obj = ToolJson.model_validate_json(tool_json_path.read_text(encoding="utf-8"))
        if tool_obj.impl.kind != "python":
            raise NotImplementedError("Skeleton supports python tools only. Add MCP support next.")

        if tool_id == "filesystem.read_text":
            func = fs_adapter.read_text
        elif tool_id == "filesystem.write_text":
            func = fs_adapter.write_text
        else:
            func = import_symbol(f"{tool_obj.impl.module}:{tool_obj.impl.function}")

        # Register function as a tool.
        # The FunctionToolset will derive schema from type hints / docstring.
        # You can enforce consistency with tool.json by adding checks here.
        toolset.add_function(func=func, name=tool_obj.name, description=tool_obj.description)

    return toolset
