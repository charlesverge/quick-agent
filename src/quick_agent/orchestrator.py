"""Helper for running agents."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import openai

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import InputAdaptor
from quick_agent.quick_agent import QuickAgent
from quick_agent.types import AgentResult


class Orchestrator:
    def __init__(
        self,
        agent_roots: list[Path] | None = None,
        tool_roots: list[Path] | None = None,
        safe_dir: Optional[Path] = None,
    ) -> None:
        self.registry: AgentRegistry = AgentRegistry(agent_roots or [])
        self.tools: AgentTools = AgentTools(tool_roots or [])
        self.directory_permissions: DirectoryPermissions = DirectoryPermissions(
            safe_dir
        )
        self.agent : QuickAgent | None = None

    async def run(
        self,
        agent_id: str,
        input_data: InputAdaptor | Path,
        extra_tools: list[str] | None = None,
        record_http_traffic: bool = False,
        enable_llm_request_logging: bool = False,
        llm_log_path: Path | str | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> AgentResult:
        self.agent = agent = QuickAgent(
            registry=self.registry,
            tools=self.tools,
            directory_permissions=self.directory_permissions,
            agent_id=agent_id,
            input_data=input_data,
            extra_tools=extra_tools,
            record_http_traffic=record_http_traffic,
            enable_llm_request_logging=enable_llm_request_logging,
            llm_log_path=llm_log_path,
            client=client,
        )
        return await agent.run()
