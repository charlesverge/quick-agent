"""Tool discovery and loading."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import JsonValue
from pydantic_ai.toolsets import FunctionToolset

from quick_agent.agent_tool_schema import (
    strip_agent_state_from_schema,
    takes_agent_state,
)
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.models.batch_request import BatchToolDefinition
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


def _discover_tool_index(tool_roots: list[Path]) -> dict[str, ToolJson]:
    index: dict[str, ToolJson] = {}
    for root in tool_roots:
        if not root.exists():
            continue
        for tool_json_path in root.rglob("tool.json"):
            raw = tool_json_path.read_text(encoding="utf-8")
            tool_obj = ToolJson.model_validate_json(raw)
            if "." in tool_obj.name:
                raise ValueError(f"Tool name {tool_obj.name!r} must not contain '.'.")
            if not tool_obj.impl.module or not tool_obj.impl.function:
                raise ValueError(
                    f"tool.json at {tool_json_path} is missing required impl fields."
                    f" Expected impl.module and impl.function, got"
                    f" module={tool_obj.impl.module!r}, function={tool_obj.impl.function!r}"
                )
            if tool_obj.impl.kind != "python":
                raise ValueError(
                    f"tool.json at {tool_json_path} has unsupported impl.kind={tool_obj.impl.kind!r}."
                    f" Only 'python' is supported."
                )
            if tool_obj.name in index:
                continue
            index[tool_obj.name] = tool_obj
    return index


def load_tool_definitions(
    tool_roots: list[Path],
    tool_names: list[str],
) -> list[BatchToolDefinition]:
    tool_index = _discover_tool_index(tool_roots)
    adapter_methods: dict[str, Callable[..., Any]] = {
        "filesystem_read_text": FilesystemToolAdapter.read_text,
        "filesystem_write_text": FilesystemToolAdapter.write_text,
        "filesystem_append_text": FilesystemToolAdapter.append_text,
        "filesystem_list_files": FilesystemToolAdapter.list_files,
        "filesystem_delete_file": FilesystemToolAdapter.delete_file,
        "filesystem_find_closest_file": FilesystemToolAdapter.find_closest_file,
        "shell_run": ShellToolAdapter.run,
    }
    result: list[BatchToolDefinition] = []
    for tool_name in tool_names:
        tool_obj = tool_index.get(tool_name)
        if tool_obj is None:
            raise FileNotFoundError(
                f"Missing tool.json for tool {tool_name} in roots: {tool_roots}"
            )
        adapter_method = adapter_methods.get(tool_name)
        if adapter_method is not None:
            func = adapter_method
        else:
            func = import_symbol(f"{tool_obj.impl.module}:{tool_obj.impl.function}")
        toolset = FunctionToolset()
        toolset.add_function(
            func=func, name=tool_name, description=tool_obj.description
        )
        tool = toolset.tools[tool_name]
        schema: dict[str, JsonValue] = tool.function_schema.json_schema
        if adapter_method is not None:
            schema = _strip_self_from_schema(schema)
        if takes_agent_state(func):
            schema = strip_agent_state_from_schema(schema)
        result.append(
            BatchToolDefinition(
                name=tool_name,
                description=tool_obj.description,
                input_schema=schema,
            )
        )
    return result


def _strip_self_from_schema(schema: dict[str, JsonValue]) -> dict[str, JsonValue]:
    schema = dict(schema)
    properties = schema.get("properties")
    if isinstance(properties, dict) and "self" in properties:
        properties = dict(properties)
        del properties["self"]
        schema["properties"] = properties
    required = schema.get("required")
    if isinstance(required, list) and "self" in required:
        schema["required"] = [r for r in required if r != "self"]
    return schema


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
        tool_obj = tool_index.get(tool_name)
        if tool_obj is None:
            raise FileNotFoundError(
                f"Missing tool.json for tool {tool_name} in roots: {tool_roots}"
            )

        if tool_obj.impl.kind != "python":
            raise NotImplementedError(
                "Skeleton supports python tools only. Add MCP support next."
            )

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

        toolset.add_function(
            func=func, name=tool_obj.name, description=tool_obj.description
        )

    return toolset
