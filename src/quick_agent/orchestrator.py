"""Helper for running agents."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import InputAdaptor
from quick_agent.quick_agent import QuickAgent


class Orchestrator:
    def __init__(
        self,
        agent_roots: list[Path] | None = None,
        tool_roots: list[Path] | None = None,
        safe_dir: Optional[Path] = None,
    ) -> None:
        self.registry: AgentRegistry = AgentRegistry(agent_roots or [])
        self.tools: AgentTools = AgentTools(tool_roots or [])
        self.directory_permissions: DirectoryPermissions = DirectoryPermissions(safe_dir)

    async def run(
        self,
        agent_id: str,
        input_data: InputAdaptor | Path,
        extra_tools: list[str] | None = None,
    ) -> BaseModel | str:
        agent = QuickAgent(
            registry=self.registry,
            tools=self.tools,
            directory_permissions=self.directory_permissions,
            agent_id=agent_id,
            input_data=input_data,
            extra_tools=extra_tools,
        )
        return await agent.run()
