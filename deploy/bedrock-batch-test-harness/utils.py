"""Shared utility functions for bedrock batch test harness stages."""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger("bedrock_batch_test_harness")


def _aws_args(
    args: list[str], *, profile: str | None = None, region: str | None = None
) -> list[str]:
    command: list[str] = ["aws"]
    if profile:
        command.extend(["--profile", profile])
    if region:
        command.extend(["--region", region])
    command.extend(args)
    return command


def run_aws(
    args: list[str], *, profile: str | None = None, region: str | None = None
) -> None:
    command = _aws_args(args, profile=profile, region=region)
    logger.info(
        f"execution: aws command > {' '.join(shlex.quote(item) for item in command)}"
    )
    subprocess.run(command, check=True)


def run_aws_json(
    args: list[str], *, profile: str | None = None, region: str | None = None
) -> dict[str, object]:
    command = _aws_args(args, profile=profile, region=region)
    logger.info(
        f"execution: aws command > {' '.join(shlex.quote(item) for item in command)}"
    )
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        return {}
    return json.loads(stdout)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False)
            file_obj.write(line)
            file_obj.write("\n")
