"""Toolset builder and agent-call injection."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel
from pydantic_ai.toolsets import FunctionToolset

from quick_agent.agent_call_tool import AgentCallTool
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import InputAdaptor
from quick_agent.tools_loader import load_tools


class AgentTools:
    def __init__(self, tool_roots: list[Path]) -> None:
        self._tool_roots: list[Path] = tool_roots

    def build_toolset(self, tool_ids: list[str], permissions: DirectoryPermissions) -> FunctionToolset[Any]:
        tool_ids_for_disk = [tool_id for tool_id in tool_ids if tool_id != "agent.call"]
        if tool_ids_for_disk:
            return load_tools(self._tool_roots, tool_ids_for_disk, permissions)
        return FunctionToolset()

    def maybe_inject_agent_call(
        self,
        tool_ids: list[str],
        toolset: FunctionToolset[Any],
        run_input_source_path: str,
        call_agent: Callable[[str, InputAdaptor | Path], Awaitable[BaseModel | str]],
    ) -> None:
        if "agent.call" not in tool_ids:
            return

        tool = AgentCallTool(call_agent, run_input_source_path)
        toolset.add_function(
            func=tool.__call__,
            name="agent_call",
            description="Call another agent and return its output.",
        )
