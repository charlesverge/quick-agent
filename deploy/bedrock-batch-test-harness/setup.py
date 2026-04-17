"""Setup stage for bedrock batch test harness."""

from __future__ import annotations

import shutil

import anyio
from settings import (
    HarnessSettings,
    resolve_runtime_settings,
    write_runtime_settings,
)
from utils import run_aws, write_jsonl

from quick_agent.agent_tools import AgentTools
from quick_agent.input_adaptors import TextInput
from quick_agent.models.batch_request import BatchSubmitRequest
from quick_agent.orchestrator import Orchestrator
from quick_agent.quick_agent import QuickAgent


async def _build_requests(
    settings: HarnessSettings, reserved: int = 0
) -> list[BatchSubmitRequest]:
    tools_roots = [settings.tools_dir, settings.repo_root / "examples"]
    limit = settings.count - reserved
    if limit < 5:
        raise ValueError(
            "Harness count must be at least 5 for chain-agent, file-manager, agent-memory, and tool-choice coverage."
        )
    orchestrator = Orchestrator([settings.agents_dir], tools_roots, safe_dir=settings.safe_dir)
    agent_memory_tools = AgentTools([settings.repo_root / "examples"])
    requests: list[BatchSubmitRequest] = []
    index = 0
    while index < limit:
        input_text = settings.input_template.format(i=index)
        if index == 1:
            request = await orchestrator.batch(
                settings.chain_agent_id, TextInput(input_text)
            )
        elif index == 2:
            request = await orchestrator.batch(
                settings.file_manager_agent_id, TextInput(settings.file_manager_input)
            )
        elif index == settings.agent_memory_index:
            agent = QuickAgent(
                registry=orchestrator.registry,
                tools=agent_memory_tools,
                directory_permissions=orchestrator.directory_permissions,
                agent_id=settings.agent_memory_agent_id,
                input_data=TextInput(settings.probe_input),
                extra_tools=None,
                memory={"first_name": settings.agent_memory_first_name},
            )
            request = agent.batch()
        elif index == settings.tool_choice_index:
            request = await orchestrator.batch(
                settings.tool_choice_agent_id,
                TextInput(settings.probe_input),
            )
        else:
            request = await orchestrator.batch(settings.agent, TextInput(input_text))
        requests.append(request)
        index += 1
    return requests


async def generate(settings: HarnessSettings, reserved: int = 0) -> None:
    requests = await _build_requests(settings, reserved)
    rows: list[dict[str, object]] = []
    submit_rows: list[dict[str, object]] = []
    for request in requests:
        request.validate_bedrock_model(settings.model_id)
        rows.append(request.jsonl_line)
        submit_rows.append(request.model_dump(mode="json"))
    write_jsonl(settings.input_jsonl, rows)
    write_jsonl(settings.submit_requests_jsonl, submit_rows)


def upload_input(settings: HarnessSettings) -> None:
    run_aws(
        ["s3", "cp", str(settings.input_jsonl), settings.s3_input_uri],
        profile=settings.aws_profile,
        region=settings.region,
    )


def run(settings: HarnessSettings, reserved: int = 0) -> HarnessSettings:
    if not settings.terraform_dir.exists():
        raise ValueError(f"Terraform directory not found: {settings.terraform_dir}")
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    if settings.safe_dir.exists():
        shutil.rmtree(settings.safe_dir)
    shutil.copytree(
        settings.harness_root / "fixtures" / "file-manager-dir", settings.safe_dir
    )
    resolved = resolve_runtime_settings(settings)
    write_runtime_settings(resolved)
    anyio.run(generate, resolved, reserved)
    return resolved
