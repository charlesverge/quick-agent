"""Setup stage for bedrock batch test harness."""

from __future__ import annotations

import anyio

from quick_agent.input_adaptors import TextInput
from quick_agent.models.batch_request import BatchSubmitRequest
from quick_agent.orchestrator import Orchestrator
from settings import HarnessSettings
from settings import resolve_runtime_settings
from settings import write_runtime_settings
from utils import run_aws
from utils import write_jsonl

CHAIN_AGENT_ID = "harness-language-chain-extractor"


async def _build_requests(settings: HarnessSettings) -> list[BatchSubmitRequest]:
    if settings.count < 2:
        raise ValueError("Harness count must be at least 2 for chain-agent coverage.")
    orchestrator = Orchestrator(
        [settings.agents_dir], [settings.tools_dir], safe_dir=settings.safe_dir
    )
    requests: list[BatchSubmitRequest] = []
    index = 0
    while index < settings.count:
        input_text = settings.input_template.format(i=index)
        if index == 1:
            request = await orchestrator.batch(CHAIN_AGENT_ID, TextInput(input_text))
        else:
            request = await orchestrator.batch(settings.agent, TextInput(input_text))
        requests.append(request)
        index += 1
    return requests


async def generate(settings: HarnessSettings) -> None:
    requests = await _build_requests(settings)
    rows: list[dict[str, object]] = []
    submit_rows: list[dict[str, object]] = []
    for request in requests:
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


def run(settings: HarnessSettings) -> HarnessSettings:
    if not settings.terraform_dir.exists():
        raise ValueError(f"Terraform directory not found: {settings.terraform_dir}")
    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    (settings.safe_dir / "bedrock").mkdir(parents=True, exist_ok=True)
    resolved = resolve_runtime_settings(settings)
    write_runtime_settings(resolved)
    anyio.run(generate, resolved)
    upload_input(resolved)
    return resolved
