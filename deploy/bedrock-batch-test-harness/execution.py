"""Execution stage for bedrock batch test harness."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from uuid import uuid4

import anyio
from pydantic import JsonValue
from settings import HarnessSettings
from utils import run_aws, run_aws_json, write_jsonl

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import TextInput
from quick_agent.models.batch_request import (
    BatchImportRequest,
    BatchMessage,
    BatchSubmitRequest,
)
from quick_agent.quick_agent import QuickAgent

logger = logging.getLogger("bedrock_batch_test_harness")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3 uri, got: {uri}")
    without_scheme = uri[len("s3://") :]
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    key = "" if len(parts) == 1 else parts[1]
    if not bucket:
        raise ValueError(f"Invalid s3 uri bucket: {uri}")
    return bucket, key


def _extract_bedrock_output_text(
    record_id: str, model_output: dict[str, object]
) -> str:
    content_obj = model_output.get("content")
    if isinstance(content_obj, list):
        text_items: list[str] = []
        for item in content_obj:
            if not isinstance(item, dict):
                continue
            if "text" not in item:
                continue
            text = item.get("text")
            if isinstance(text, str):
                text_items.append(text)
        if text_items:
            return "".join(text_items)
    choices_obj = model_output.get("choices")
    if isinstance(choices_obj, list):
        index = 0
        while index < len(choices_obj):
            choice_obj = choices_obj[index]
            if isinstance(choice_obj, dict):
                message_obj = choice_obj.get("message")
                if isinstance(message_obj, dict):
                    content_value = message_obj.get("content")
                    if isinstance(content_value, str) and content_value:
                        return content_value
            index += 1
    raise ValueError(
        f"Bedrock result missing supported output fields for record_id={record_id} (modelOutput.content or modelOutput.choices[].message.content)."
    )


def _create_job(
    *,
    model_id: str,
    role_arn: str,
    s3_input_uri: str,
    s3_output_uri: str,
    job_name: str,
    region: str,
    aws_profile: str,
) -> str:
    response = run_aws_json(
        [
            "--region",
            region,
            "bedrock",
            "create-model-invocation-job",
            "--model-id",
            model_id,
            "--role-arn",
            role_arn,
            "--job-name",
            job_name,
            "--input-data-config",
            json.dumps({"s3InputDataConfig": {"s3Uri": s3_input_uri}}),
            "--output-data-config",
            json.dumps({"s3OutputDataConfig": {"s3Uri": s3_output_uri}}),
        ],
        profile=aws_profile,
    )
    job_arn = response.get("jobArn")
    if not isinstance(job_arn, str):
        raise ValueError("Bedrock create-model-invocation-job did not return jobArn.")
    return job_arn


def _wait_for_job(
    *,
    job_arn: str,
    region: str,
    poll_seconds: int,
    timeout_seconds: int,
    aws_profile: str,
) -> dict[str, object]:
    start = time.time()
    poll_count = 0
    while True:
        poll_count += 1
        response = run_aws_json(
            [
                "--region",
                region,
                "bedrock",
                "get-model-invocation-job",
                "--job-identifier",
                job_arn,
            ],
            profile=aws_profile,
        )
        status = response.get("status")
        elapsed = time.time() - start
        logger.info(
            f"execution: poll status > job_arn={job_arn} poll={poll_count} status={status} elapsed_seconds={int(elapsed)}"
        )
        if status == "Completed":
            return response
        if status in ("Failed", "Stopped", "PartiallyCompleted"):
            raise RuntimeError(f"Bedrock job ended with status={status}: {response}")
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"Timed out waiting for Bedrock job completion after {timeout_seconds}s."
            )
        time.sleep(poll_seconds)


def _job_id_from_arn(job_arn: str) -> str:
    if "/" not in job_arn:
        raise ValueError(f"Unable to parse job id from arn: {job_arn}")
    job_id = job_arn.rsplit("/", 1)[-1]
    if not job_id:
        raise ValueError(f"Unable to parse job id from arn: {job_arn}")
    return job_id


def _expected_output_jsonl_uri(
    *, s3_output_uri: str, input_jsonl: Path, job_id: str
) -> str:
    if not s3_output_uri.endswith("/"):
        s3_output_uri = f"{s3_output_uri}/"
    return f"{s3_output_uri}{job_id}/{input_jsonl.name}.out"


def _assert_output_exists(*, output_s3_uri: str, region: str, aws_profile: str) -> None:
    bucket, key = _parse_s3_uri(output_s3_uri)
    if not key:
        raise ValueError(f"Expected output object key in s3 uri: {output_s3_uri}")
    response = run_aws_json(
        [
            "--region",
            region,
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            key,
        ],
        profile=aws_profile,
    )
    if not isinstance(response, dict):
        raise ValueError(f"Unable to verify expected output object: {output_s3_uri}")


def _find_output_jsonl(
    *,
    s3_output_uri: str,
    region: str,
    aws_profile: str,
    input_jsonl: Path,
    job_id: str,
) -> str:
    expected_uri = _expected_output_jsonl_uri(
        s3_output_uri=s3_output_uri,
        input_jsonl=input_jsonl,
        job_id=job_id,
    )
    _assert_output_exists(
        output_s3_uri=expected_uri,
        region=region,
        aws_profile=aws_profile,
    )
    return expected_uri


def _download_output_jsonl(
    *,
    output_s3_uri: str,
    local_output_path: Path,
    region: str,
    aws_profile: str,
) -> None:
    local_output_path.parent.mkdir(parents=True, exist_ok=True)
    run_aws(
        ["s3", "cp", output_s3_uri, str(local_output_path)],
        profile=aws_profile,
        region=region,
    )


def _run_batch_job(
    *,
    settings: HarnessSettings,
    input_s3_uri: str,
    input_jsonl_path: Path,
    output_path: Path,
    job_name: str,
) -> list[dict[str, object]]:
    logger.info(
        f"execution: submitting Bedrock batch job > model_id={settings.model_id} job_name={job_name}"
    )
    job_arn = _create_job(
        model_id=settings.model_id,
        role_arn=settings.role_arn,
        s3_input_uri=input_s3_uri,
        s3_output_uri=settings.s3_output_uri,
        job_name=job_name,
        region=settings.region,
        aws_profile=settings.aws_profile,
    )
    logger.info(
        f"execution: waiting for Bedrock batch job completion > job_arn={job_arn}"
    )
    completed = _wait_for_job(
        job_arn=job_arn,
        region=settings.region,
        poll_seconds=settings.poll_seconds,
        timeout_seconds=settings.timeout_seconds,
        aws_profile=settings.aws_profile,
    )
    output_cfg = completed.get("outputDataConfig")
    if not isinstance(output_cfg, dict):
        raise ValueError("Bedrock get-model-invocation-job missing outputDataConfig.")
    s3_output_cfg = output_cfg.get("s3OutputDataConfig")
    if not isinstance(s3_output_cfg, dict):
        raise ValueError("Bedrock get-model-invocation-job missing s3OutputDataConfig.")
    s3_output_uri = s3_output_cfg.get("s3Uri")
    if not isinstance(s3_output_uri, str):
        raise ValueError("Bedrock get-model-invocation-job missing output s3Uri.")
    job_id = _job_id_from_arn(job_arn)
    jsonl_uri = _find_output_jsonl(
        s3_output_uri=s3_output_uri,
        region=settings.region,
        aws_profile=settings.aws_profile,
        input_jsonl=input_jsonl_path,
        job_id=job_id,
    )
    logger.info(f"execution: downloading Bedrock output jsonl > s3_uri={jsonl_uri}")
    _download_output_jsonl(
        output_s3_uri=jsonl_uri,
        local_output_path=output_path,
        region=settings.region,
        aws_profile=settings.aws_profile,
    )
    return _load_jsonl(output_path)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line_value = line.strip()
            if not line_value:
                continue
            item = json.loads(line_value)
            if not isinstance(item, dict):
                raise ValueError("Expected JSON object row in jsonl.")
            rows.append(item)
    return rows


def _has_tool_use_anthropic(model_output: dict[str, object]) -> bool:
    content = model_output.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and item.get("type") == "tool_use" for item in content
    )


def _has_tool_use_converse(model_output: dict[str, object]) -> bool:
    content = model_output.get("content")
    if not isinstance(content, list):
        return False
    return any(isinstance(item, dict) and "toolUse" in item for item in content)


def _has_tool_use_openai(model_output: dict[str, object]) -> bool:
    choices = model_output.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                return True
    return False


def _has_tool_use(model_output: dict[str, object]) -> bool:
    return (
        _has_tool_use_anthropic(model_output)
        or _has_tool_use_converse(model_output)
        or _has_tool_use_openai(model_output)
    )


def _raw_tool_calls(model_output: dict[str, object]) -> list[JsonValue]:
    calls: list[JsonValue] = []
    content = model_output.get("content")
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and (
                item.get("type") == "tool_use" or "toolUse" in item
            ):
                calls.append(item)
    choices = model_output.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message")
            if not isinstance(message, dict):
                continue
            tool_calls_list = message.get("tool_calls")
            if isinstance(tool_calls_list, list):
                calls.extend(tool_calls_list)
    return calls


def _to_import_request(row: dict[str, object]) -> BatchImportRequest:
    record_id = row.get("recordId")
    if not isinstance(record_id, str):
        raise ValueError("Bedrock row missing string recordId.")
    if "error" in row:
        error_obj = row.get("error")
        if isinstance(error_obj, dict):
            message_obj = error_obj.get("message")
            if isinstance(message_obj, str):
                return BatchImportRequest(
                    request_id=record_id,
                    payload={"state": "error", "message": message_obj},
                )
            error_message_obj = error_obj.get("errorMessage")
            if isinstance(error_message_obj, str):
                return BatchImportRequest(
                    request_id=record_id,
                    payload={"state": "error", "message": error_message_obj},
                )
        raise ValueError(f"Bedrock error row missing error.message/errorMessage: {row}")
    model_output_obj = row.get("modelOutput")
    if not isinstance(model_output_obj, dict):
        raise ValueError("Bedrock row missing modelOutput object.")
    if _has_tool_use(model_output_obj):
        tool_calls: list[JsonValue] = _raw_tool_calls(model_output_obj)
        return BatchImportRequest(
            request_id=record_id,
            payload={"state": "tool_use", "tool_calls": tool_calls},
        )
    output_text = _extract_bedrock_output_text(record_id, model_output_obj)
    return BatchImportRequest(
        request_id=record_id,
        payload={"state": "completed", "output": output_text},
    )


async def import_result_from_settings(settings: HarnessSettings) -> None:
    logger.info("execution: importing Bedrock output into quick-agent outcomes")
    submit_rows = _load_jsonl(settings.submit_requests_jsonl)
    current_requests: list[BatchSubmitRequest] = []
    index = 0
    while index < len(submit_rows):
        current_requests.append(BatchSubmitRequest.model_validate(submit_rows[index]))
        index += 1
    all_submit_requests: list[BatchSubmitRequest] = list(current_requests)
    root_ids: dict[str, str] = {}
    root_order: list[str] = []
    for request in current_requests:
        root_ids[request.request_id] = request.request_id
        root_order.append(request.request_id)
    if not current_requests:
        raise ValueError("No submit requests found for execution stage.")
    padding_template: BatchSubmitRequest | None = None
    for request in current_requests:
        if request.agent_id == settings.agent:
            padding_template = request
            break
    if padding_template is None:
        raise ValueError(
            f"Unable to find padding template request for agent_id={settings.agent}"
        )
    registry = AgentRegistry([settings.agents_dir])
    tools = AgentTools([settings.tools_dir, settings.repo_root / "examples"])
    directory_permissions = DirectoryPermissions(settings.safe_dir)
    final_outcomes: dict[str, dict[str, object]] = {}
    all_output_rows: list[dict[str, object]] = []
    round_index = 1
    while current_requests:
        current_output_path = settings.runtime_dir / f"output-round-{round_index}.jsonl"
        if round_index == 1:
            round1_output_path = current_output_path
            if round1_output_path.exists():
                rows = _load_jsonl(round1_output_path)
                cached_ids = {r.get("recordId") for r in rows}
                submit_ids = {req.request_id for req in current_requests}
                if cached_ids != submit_ids:
                    raise ValueError(
                        f"Round 1 output recordIds do not match current submit requests. "
                        f"Delete {round1_output_path} and re-run."
                    )
                logger.info(
                    f"execution: reusing existing round 1 output > path={round1_output_path}"
                )
            else:
                rows = _run_batch_job(
                    settings=settings,
                    input_s3_uri=settings.s3_input_uri,
                    input_jsonl_path=settings.input_jsonl,
                    output_path=round1_output_path,
                    job_name=settings.job_name,
                )
        else:
            round_input_name = f"input-round-{round_index}.jsonl"
            round_input_path = settings.runtime_dir / round_input_name
            round_output_path = (
                settings.runtime_dir / f"output-round-{round_index}.jsonl"
            )
            submit_round_path = (
                settings.runtime_dir / f"submit-round-{round_index}.jsonl"
            )
            bucket, key = _parse_s3_uri(settings.s3_input_uri)
            key_prefix = key.rsplit("/", 1)[0] if "/" in key else ""
            if key_prefix:
                round_s3_uri = f"s3://{bucket}/{key_prefix}/{round_input_name}"
            else:
                round_s3_uri = f"s3://{bucket}/{round_input_name}"
            if round_output_path.exists():
                if not round_input_path.exists():
                    raise ValueError(
                        f"Round {round_index} output exists but input file is missing: {round_input_path}"
                    )
                cached_input_rows = _load_jsonl(round_input_path)
                cached_output_rows = _load_jsonl(round_output_path)
                input_ids = {r.get("recordId") for r in cached_input_rows}
                output_ids = {r.get("recordId") for r in cached_output_rows}
                if input_ids != output_ids:
                    raise ValueError(
                        f"Round {round_index} output recordIds do not match input recordIds. "
                        f"Delete {round_output_path} and re-run."
                    )
                logger.info(
                    f"execution: reusing existing round {round_index} output > path={round_output_path}"
                )
                if submit_round_path.exists():
                    submitted_requests: list[BatchSubmitRequest] = [
                        BatchSubmitRequest.model_validate(r)
                        for r in _load_jsonl(submit_round_path)
                    ]
                    for req in submitted_requests:
                        if req.request_id not in root_ids:
                            root_ids[req.request_id] = req.request_id
                else:
                    logger.info(
                        f"execution: reconstructing round {round_index} submit requests from cached input > path={round_input_path}"
                    )
                    padded: list[BatchSubmitRequest] = list(current_requests)
                    while len(padded) < len(cached_input_rows):
                        pad_req = padding_template.model_copy(
                            update={"request_id": f"noop-{settings.agent}-{uuid4()}"}
                        )
                        padded.append(pad_req)
                    if len(padded) != len(cached_input_rows):
                        raise ValueError(
                            f"Round {round_index}: reconstructed request count ({len(padded)}) "
                            f"does not match cached input row count ({len(cached_input_rows)})"
                        )
                    submitted_requests = []
                    for pos, req in enumerate(padded):
                        record_id_val = cached_input_rows[pos].get("recordId")
                        if not isinstance(record_id_val, str):
                            raise ValueError(
                                f"Round {round_index}: cached input row {pos} missing recordId"
                            )
                        old_id = req.request_id
                        new_req = req.model_copy(update={"request_id": record_id_val})
                        if old_id in root_ids:
                            root_ids[record_id_val] = root_ids[old_id]
                        else:
                            root_ids[record_id_val] = record_id_val
                        submitted_requests.append(new_req)
                rows = cached_output_rows
            else:
                if round_input_path.exists():
                    logger.info(
                        f"execution: reusing existing round {round_index} input > path={round_input_path}"
                    )
                    if not submit_round_path.exists():
                        raise ValueError(
                            f"Round {round_index} submit requests file is missing: {submit_round_path}. "
                            f"Delete round {round_index} input and re-run."
                        )
                    submitted_requests = [
                        BatchSubmitRequest.model_validate(r)
                        for r in _load_jsonl(submit_round_path)
                    ]
                    for req in submitted_requests:
                        if req.request_id not in root_ids:
                            root_ids[req.request_id] = req.request_id
                else:
                    submitted_requests = list(current_requests)
                    if len(submitted_requests) < settings.count:
                        logger.info(
                            f"execution: padding round {round_index} requests from {len(submitted_requests)} to {settings.count}"
                        )
                    while len(submitted_requests) < settings.count:
                        padded_request = padding_template.model_copy(
                            update={
                                "request_id": f"noop-{settings.agent}-{uuid4()}",
                                "messages": [
                                    BatchMessage(role="user", content="say ok")
                                ],
                                "tool_use_enabled": False,
                                "tool_choice": None,
                                "tool_ids": [],
                                "tools": None,
                                "response_format": None,
                                "max_tool_calls": 1,
                                "context": padding_template.context.model_copy(
                                    update={"input_text": "say ok", "extra_tools": []}
                                ),
                            }
                        )
                        root_ids[padded_request.request_id] = padded_request.request_id
                        submitted_requests.append(padded_request)
                    round_rows: list[dict[str, object]] = []
                    for request in submitted_requests:
                        round_rows.append(request.jsonl_line)
                    write_jsonl(round_input_path, round_rows)
                    write_jsonl(
                        submit_round_path,
                        [req.model_dump(mode="json") for req in submitted_requests],
                    )
                logger.info(f"execution: uploading round input > s3_uri={round_s3_uri}")
                run_aws(
                    ["s3", "cp", str(round_input_path), round_s3_uri],
                    profile=settings.aws_profile,
                    region=settings.region,
                )
                round_job_name = f"{settings.job_name}-r{round_index}"
                rows = _run_batch_job(
                    settings=settings,
                    input_s3_uri=round_s3_uri,
                    input_jsonl_path=round_input_path,
                    output_path=round_output_path,
                    job_name=round_job_name,
                )
            all_submit_requests.extend(submitted_requests)
            current_requests = submitted_requests
        all_output_rows.extend(rows)
        current_index: dict[str, BatchSubmitRequest] = {}
        for request in current_requests:
            current_index[request.request_id] = request
        next_requests: list[BatchSubmitRequest] = []
        seen_ids: set[str] = set()
        row_index = 0
        while row_index < len(rows):
            row = rows[row_index]
            request_id = "unknown"
            agent_id = "unknown"
            try:
                batch_import = _to_import_request(row)
                request_id = batch_import.request_id
                submit_request = current_index.get(request_id)
                if submit_request is None:
                    raise ValueError(
                        f"Missing submit request context for recordId={request_id}"
                    )
                agent_id = submit_request.agent_id
                seen_ids.add(request_id)
                if request_id.startswith("noop-"):
                    row_index += 1
                    continue
                context = submit_request.context
                agent = QuickAgent(
                    registry=registry,
                    tools=tools,
                    directory_permissions=directory_permissions,
                    agent_id=agent_id,
                    input_data=TextInput(context.input_text),
                    extra_tools=context.extra_tools,
                    memory=dict(context.memory),
                )
                agent.load_batch_context(context=context)
                batch_import_to_use = batch_import
                if batch_import.payload.get("state") == "tool_use":
                    submit_dump: dict[str, JsonValue] = submit_request.model_dump(
                        mode="json"
                    )
                    augmented_payload: dict[str, JsonValue] = dict(batch_import.payload)
                    augmented_payload["submit_request"] = submit_dump
                    batch_import_to_use = BatchImportRequest(
                        request_id=request_id,
                        provider_job_id=batch_import.provider_job_id,
                        payload=augmented_payload,
                    )
                outcome = await agent.import_result(batch_import=batch_import_to_use)
            except Exception as err:
                raise ValueError(
                    f"agent={agent_id}"
                    f" request_id={request_id}"
                    f" round={round_index} row={row_index}"
                    f" file={current_output_path}:{row_index + 1}"
                    f" > {err}"
                ) from err
            if outcome.next_request is not None:
                next_request = outcome.next_request
                root_id = root_ids[submit_request.request_id]
                root_ids[next_request.request_id] = root_id
                next_requests.append(next_request)
            else:
                root_id = root_ids[submit_request.request_id]
                final_outcomes[root_id] = outcome.model_dump(mode="json")
            row_index += 1
        if len(seen_ids) != len(current_requests):
            raise ValueError(
                f"Round {round_index} output row count mismatch: expected={len(current_requests)} actual={len(seen_ids)}"
            )
        if next_requests:
            all_submit_requests.extend(next_requests)
        current_requests = next_requests
        round_index += 1
    ordered_outcomes: list[dict[str, object]] = []
    for root_id in root_order:
        outcome_obj = final_outcomes.get(root_id)
        if outcome_obj is None:
            raise ValueError(f"Missing final outcome for initial request_id={root_id}")
        ordered_outcomes.append(outcome_obj)
    submit_dump_rows: list[dict[str, object]] = []
    submit_ids: set[str] = set()
    for request in all_submit_requests:
        if request.request_id in submit_ids:
            continue
        submit_ids.add(request.request_id)
        submit_dump_rows.append(request.model_dump(mode="json"))
    write_jsonl(settings.submit_requests_jsonl, submit_dump_rows)
    write_jsonl(settings.output_jsonl, all_output_rows)
    write_jsonl(settings.outcomes_jsonl, ordered_outcomes)


def run(settings: HarnessSettings) -> None:
    anyio.run(import_result_from_settings, settings)
