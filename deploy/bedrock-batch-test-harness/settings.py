"""Shared settings for the bedrock batch test harness."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class HarnessSettings:
    repo_root: Path
    harness_root: Path
    terraform_dir: Path
    logs_dir: Path
    runtime_dir: Path
    fixture_name: str
    agent: str
    agents_dir: Path
    tools_dir: Path
    safe_dir: Path
    aws_profile: str
    model_id: str
    region: str
    role_arn: str
    s3_input_uri: str
    s3_output_uri: str
    input_template: str
    count: int
    poll_seconds: int
    timeout_seconds: int
    probe_input: str
    job_name: str
    submit_requests_jsonl: Path
    input_jsonl: Path
    output_jsonl: Path
    outcomes_jsonl: Path
    harness_aws_profile: str = "quick-agent-bedrock-deployer"
    harness_aws_region: str = "us-east-1"
    harness_poll_seconds: int = 30
    harness_timeout_seconds: int = 36000
    chain_agent_id: str = "harness-language-chain-extractor"
    chain_first_step_id: str = "generate-random-name"
    chain_second_step_id: str = "tech-keyword-extraction"
    file_manager_agent_id: str = "harness-file-manager"
    file_manager_index: int = 2
    file_manager_directory: str = "agent_working_directory"
    file_manager_append_text: str = "text to append"
    agent_memory_agent_id: str = "harness-agent-memory"
    agent_memory_index: int = 3
    agent_memory_first_name: str = "Charles"
    file_manager_input: str = Path(__file__).resolve().parent.joinpath("fixtures/file_manager_input.json").read_text(
        encoding="utf-8"
    )

    @property
    def runtime_settings_path(self) -> Path:
        return self.runtime_dir / "runtime_settings.json"

    def to_json_dict(self) -> dict[str, object]:
        return {
            "repo_root": str(self.repo_root),
            "harness_root": str(self.harness_root),
            "terraform_dir": str(self.terraform_dir),
            "logs_dir": str(self.logs_dir),
            "runtime_dir": str(self.runtime_dir),
            "fixture_name": self.fixture_name,
            "agent": self.agent,
            "agents_dir": str(self.agents_dir),
            "tools_dir": str(self.tools_dir),
            "safe_dir": str(self.safe_dir),
            "aws_profile": self.aws_profile,
            "model_id": self.model_id,
            "region": self.region,
            "role_arn": self.role_arn,
            "s3_input_uri": self.s3_input_uri,
            "s3_output_uri": self.s3_output_uri,
            "input_template": self.input_template,
            "count": self.count,
            "poll_seconds": self.poll_seconds,
            "timeout_seconds": self.timeout_seconds,
            "probe_input": self.probe_input,
            "job_name": self.job_name,
            "submit_requests_jsonl": str(self.submit_requests_jsonl),
            "input_jsonl": str(self.input_jsonl),
            "output_jsonl": str(self.output_jsonl),
            "outcomes_jsonl": str(self.outcomes_jsonl),
        }


def _fixture_path(harness_root: Path, fixture_name: str) -> Path:
    return harness_root / "fixtures" / f"{fixture_name}.json"


def _required_str(config: dict[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Fixture value '{key}' must be a string.")
    return value


def _required_int(config: dict[str, object], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise ValueError(f"Fixture value '{key}' must be an integer.")
    return value


def load_settings(*, fixture_name: str) -> HarnessSettings:
    harness_root = Path(__file__).resolve().parent
    repo_root = harness_root.parent.parent
    fixture_file = _fixture_path(harness_root, fixture_name)
    fixture_data = json.loads(fixture_file.read_text(encoding="utf-8"))
    if not isinstance(fixture_data, dict):
        raise ValueError(f"Fixture must be a JSON object: {fixture_file}")
    job_name_prefix = _required_str(fixture_data, "job_name_prefix")
    job_name = f"{job_name_prefix}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    safe_dir = harness_root / _required_str(fixture_data, "safe_dir")
    return HarnessSettings(
        repo_root=repo_root,
        harness_root=harness_root,
        terraform_dir=harness_root / "terraform",
        logs_dir=harness_root / "logs",
        runtime_dir=harness_root / "runtime",
        fixture_name=fixture_name,
        agent=_required_str(fixture_data, "agent"),
        agents_dir=repo_root / _required_str(fixture_data, "agents_dir"),
        tools_dir=repo_root / _required_str(fixture_data, "tools_dir"),
        safe_dir=safe_dir,
        aws_profile=HarnessSettings.harness_aws_profile,
        model_id=_required_str(fixture_data, "model_id"),
        region=HarnessSettings.harness_aws_region,
        role_arn=_required_str(fixture_data, "role_arn"),
        s3_input_uri=_required_str(fixture_data, "s3_input_uri"),
        s3_output_uri=_required_str(fixture_data, "s3_output_uri"),
        input_template=_required_str(fixture_data, "input_template"),
        count=_required_int(fixture_data, "count"),
        poll_seconds=HarnessSettings.harness_poll_seconds,
        timeout_seconds=HarnessSettings.harness_timeout_seconds,
        probe_input=_required_str(fixture_data, "probe_input"),
        job_name=job_name,
        submit_requests_jsonl=harness_root / "runtime" / "submit-requests-100.jsonl",
        input_jsonl=harness_root / "runtime" / "input-100.jsonl",
        output_jsonl=harness_root / "runtime" / "output-100.jsonl",
        outcomes_jsonl=harness_root / "runtime" / "import-outcomes-100.jsonl",
    )


def _terraform_output(terraform_dir: Path, key: str) -> str:
    completed = subprocess.run(
        ["terraform", f"-chdir={terraform_dir}", "output", "-raw", key],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def resolve_runtime_settings(settings: HarnessSettings) -> HarnessSettings:
    model_id = settings.model_id or _terraform_output(
        settings.terraform_dir, "model_id"
    )
    role_arn = settings.role_arn or _terraform_output(
        settings.terraform_dir, "bedrock_batch_role_arn"
    )
    s3_input_uri = settings.s3_input_uri or _terraform_output(
        settings.terraform_dir, "s3_input_uri"
    )
    s3_output_uri = settings.s3_output_uri or _terraform_output(
        settings.terraform_dir, "s3_output_uri"
    )
    return replace(
        settings,
        model_id=model_id,
        role_arn=role_arn,
        s3_input_uri=s3_input_uri,
        s3_output_uri=s3_output_uri,
    )


def write_runtime_settings(settings: HarnessSettings) -> None:
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    payload = settings.to_json_dict()
    settings.runtime_settings_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def load_runtime_settings(*, harness_root: Path) -> HarnessSettings:
    runtime_file = harness_root / "runtime" / "runtime_settings.json"
    payload = json.loads(runtime_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Runtime settings must be a JSON object: {runtime_file}")
    values: dict[str, object] = payload
    return HarnessSettings(
        repo_root=Path(_required_str(values, "repo_root")),
        harness_root=Path(_required_str(values, "harness_root")),
        terraform_dir=Path(_required_str(values, "terraform_dir")),
        logs_dir=Path(_required_str(values, "logs_dir")),
        runtime_dir=Path(_required_str(values, "runtime_dir")),
        fixture_name=_required_str(values, "fixture_name"),
        agent=_required_str(values, "agent"),
        agents_dir=Path(_required_str(values, "agents_dir")),
        tools_dir=Path(_required_str(values, "tools_dir")),
        safe_dir=Path(_required_str(values, "safe_dir")),
        aws_profile=_required_str(values, "aws_profile"),
        model_id=_required_str(values, "model_id"),
        region=_required_str(values, "region"),
        role_arn=_required_str(values, "role_arn"),
        s3_input_uri=_required_str(values, "s3_input_uri"),
        s3_output_uri=_required_str(values, "s3_output_uri"),
        input_template=_required_str(values, "input_template"),
        count=_required_int(values, "count"),
        poll_seconds=_required_int(values, "poll_seconds"),
        timeout_seconds=_required_int(values, "timeout_seconds"),
        probe_input=_required_str(values, "probe_input"),
        job_name=_required_str(values, "job_name"),
        submit_requests_jsonl=Path(_required_str(values, "submit_requests_jsonl")),
        input_jsonl=Path(_required_str(values, "input_jsonl")),
        output_jsonl=Path(_required_str(values, "output_jsonl")),
        outcomes_jsonl=Path(_required_str(values, "outcomes_jsonl")),
    )
