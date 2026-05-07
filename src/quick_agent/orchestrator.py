"""Helper for running agents."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import openai

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import InputAdaptor
from quick_agent.models.batch_request import (
    BatchImportOutcome,
    BatchImportRequest,
    BatchSubmitRequest,
)
from quick_agent.quick_agent import QuickAgent
from quick_agent.types import AgentResult, StepOutput


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
        self.agent: QuickAgent | None = None

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

    async def batch(
        self,
        agent_id: str,
        input_data: InputAdaptor | Path,
        extra_tools: list[str] | None = None,
        record_http_traffic: bool = False,
        enable_llm_request_logging: bool = False,
        llm_log_path: Path | str | None = None,
        client: openai.AsyncOpenAI | None = None,
        memory: dict[str, object] | None = None,
    ) -> list[BatchSubmitRequest]:
        bedrock_model = self.registry.get(agent_id).spec.model.model_copy(
            update={"provider": "bedrock"}
        )
        self.agent = agent = QuickAgent(
            registry=self.registry,
            tools=self.tools,
            directory_permissions=self.directory_permissions,
            agent_id=agent_id,
            input_data=input_data,
            extra_tools=extra_tools,
            model=bedrock_model,
            record_http_traffic=record_http_traffic,
            enable_llm_request_logging=enable_llm_request_logging,
            llm_log_path=llm_log_path,
            client=client,
            memory=memory,
        )
        return agent.batch()

    async def import_result(
        self,
        agent_id: str,
        input_data: InputAdaptor | Path,
        batch_import: BatchImportRequest,
        extra_tools: list[str] | None = None,
        record_http_traffic: bool = False,
        enable_llm_request_logging: bool = False,
        llm_log_path: Path | str | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> BatchImportOutcome:
        bedrock_model = self.registry.get(agent_id).spec.model.model_copy(
            update={"provider": "bedrock"}
        )
        self.agent = agent = QuickAgent(
            registry=self.registry,
            tools=self.tools,
            directory_permissions=self.directory_permissions,
            agent_id=agent_id,
            input_data=input_data,
            extra_tools=extra_tools,
            model=bedrock_model,
            record_http_traffic=record_http_traffic,
            enable_llm_request_logging=enable_llm_request_logging,
            llm_log_path=llm_log_path,
            client=client,
        )
        return await agent.import_result(batch_import=batch_import)

    async def batch_execute(
        self,
        agent_id: str,
        input_data: InputAdaptor | Path,
        extra_tools: list[str] | None = None,
        record_http_traffic: bool = False,
        enable_llm_request_logging: bool = False,
        llm_log_path: Path | str | None = None,
        client: openai.AsyncOpenAI | None = None,
        memory: dict[str, object] | None = None,
    ) -> AgentResult:
        agent = QuickAgent(
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
            memory=memory,
            test_mode=True,
        )
        processor = agent.processor
        if processor is None:
            raise ValueError(
                "Failed to create agent processor. Check test_mode configuration."
            )
        batch_requests = agent.batch()
        if len(batch_requests) > 1:
            items: list[AgentResult] = []
            for batch_request in batch_requests:
                import_request = await processor.run_batch(batch_request)
                outcome = await agent.import_result(batch_import=import_request)
                result = outcome.result
                if result is None:
                    raise ValueError("Batch execution produced no result.")
                items.append(result)
            return items
        import_request = await processor.run_batch(batch_requests[0])
        outcome = await agent.import_result(batch_import=import_request)

        while outcome.next_request:
            agent = QuickAgent(
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
                memory=memory,
                test_mode=True,
            )
            processor = agent.processor
            if processor is None:
                raise ValueError(
                    "Failed to create agent processor. Check test_mode configuration."
                )
            next_req = outcome.next_request
            if next_req.context and next_req.context.state:
                agent_state = next_req.context.state
                steps_value = agent_state.get("steps")
                last_output_value = agent_state.get("last_step_output")
                steps: dict[str, StepOutput] | None = None
                if isinstance(steps_value, dict):
                    steps = steps_value  # type: ignore[assignment]
                last_output: StepOutput | None = None
                if last_output_value is not None:
                    last_output = last_output_value  # type: ignore[assignment]
                agent.state = {
                    "agent_id": agent_id,
                    "steps": steps if steps is not None else {},
                    "last_step_output": last_output,
                }

            batch_requests = agent.batch()
            import_request = await processor.run_batch(batch_requests[0])
            outcome = await agent.import_result(batch_import=import_request)

        result = outcome.result
        if result is None:
            raise ValueError("Batch execution produced no result.")
        return result
