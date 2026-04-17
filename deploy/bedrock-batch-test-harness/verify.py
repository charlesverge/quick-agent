"""Verification stage for bedrock batch test harness."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from models import OutcomeRowsResult, StageResult, VerificationResult
from pydantic import ValidationError
from schemas.tech_keywords import RandomName, TechKeywords
from settings import HarnessSettings
from verify_code_rule import _parse_evaluation
from verify_csv import RequestTracker, export_csv_by_agent, export_summary_csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REQUIRED_LANGUAGES = {"python", "typescript"}
REQUIRED_DATABASES = {"mongodb", "sql"}


def _normalize(value: str) -> str:
    return value.strip().lower()


def _normalized_set(values: list[str], *, context: str) -> set[str]:
    normalized: set[str] = set()
    index = 0
    while index < len(values):
        value = _normalize(values[index])
        if not value:
            raise ValueError(f"{context}: empty string value is not allowed")
        if value in normalized:
            raise ValueError(f"{context}: duplicate value '{value}'")
        normalized.add(value)
        index += 1
    return normalized


def _parse_tech_keywords(raw: object, *, context: str) -> TechKeywords:
    try:
        if isinstance(raw, dict):
            return TechKeywords.model_validate(raw)
        if isinstance(raw, str):
            try:
                return TechKeywords.model_validate_json(raw)
            except ValidationError:
                start = raw.find("{")
                end = raw.rfind("}")
                if start >= 0 and end > start:
                    return TechKeywords.model_validate_json(raw[start : end + 1])
        raise ValueError(f"{context}: unsupported final_result type {type(raw)}")
    except ValidationError as error:
        raise ValueError(
            f"{context}: final_result is not valid TechKeywords"
        ) from error


def _validate_tech_keywords(keywords: TechKeywords, *, context: str) -> list[str]:
    warnings: list[str] = []
    languages = _normalized_set(
        keywords.computer_languages, context=f"{context} computer_languages"
    )
    databases = _normalized_set(keywords.databases, context=f"{context} databases")
    other = _normalized_set(keywords.other, context=f"{context} other")

    missing_languages = REQUIRED_LANGUAGES - languages
    if missing_languages:
        raise ValueError(
            f"{context}: missing required computer_languages values {sorted(missing_languages)} found {keywords.computer_languages}"
        )
    if not ({"node.js", "nodejs"} & languages):
        msg = f"{context}: missing required computer_languages value node.js"
        logger.warning(msg)
        warnings.append(msg)

    missing_databases = REQUIRED_DATABASES - databases
    if missing_databases:
        raise ValueError(
            f"{context}: missing required databases values {sorted(missing_databases)}"
        )

    all_terms = set()
    all_terms.update(languages)
    all_terms.update(databases)
    all_terms.update(other)
    if "react" not in all_terms:
        msg = f"{context}: missing required extracted term react"
        logger.warning(msg)
        warnings.append(msg)

    return warnings


def _require_str(row: dict[str, object], key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{context}: expected string '{key}', got {type(value)}")
    return value


def _require_dict(
    row: dict[str, object], key: str, *, context: str
) -> dict[str, object]:
    value = row.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context}: expected object '{key}', got {type(value)}")
    return value


def _validate_input_rows(
    rows: list[dict[str, object]], expected_count: int
) -> set[str]:
    if len(rows) != expected_count:
        raise ValueError(
            f"input row count mismatch: expected={expected_count}, actual={len(rows)}"
        )
    ids: set[str] = set()
    index = 0
    while index < len(rows):
        row = rows[index]
        context = f"input row index={index}"
        record_id = _require_str(row, "recordId", context=context)
        if record_id in ids:
            raise ValueError(f"{context}: duplicate recordId={record_id}")
        ids.add(record_id)
        model_input = _require_dict(row, "modelInput", context=context)
        messages = model_input.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"{context}: modelInput.messages must be a list")
        index += 1
    return ids


def _validate_output_rows(
    rows: list[dict[str, object]], expected_ids: set[str]
) -> None:
    ids: set[str] = set()
    index = 0
    while index < len(rows):
        row = rows[index]
        base_context = f"output row index={index}"
        record_id = _require_str(row, "recordId", context=base_context)
        context = f"{base_context} recordId={record_id}"
        if record_id in ids:
            raise ValueError(f"{context}: duplicate recordId={record_id}")
        ids.add(record_id)
        if "modelOutput" in row and "error" in row:
            raise ValueError(
                f"{context}: row cannot contain both modelOutput and error"
            )
        if "modelOutput" in row:
            model_output = _require_dict(row, "modelOutput", context=context)
            content = model_output.get("content")
            if isinstance(content, list):
                index += 1
                continue
            choices = model_output.get("choices")
            if not isinstance(choices, list):
                raise ValueError(
                    f"{context}: modelOutput must include either content list or choices list"
                )
            if not choices:
                raise ValueError(f"{context}: modelOutput.choices cannot be empty")
        elif "error" in row:
            error_obj = _require_dict(row, "error", context=context)
            message_obj = error_obj.get("message")
            if isinstance(message_obj, str):
                index += 1
                continue
            _ = _require_str(error_obj, "errorMessage", context=f"{context} error")
        else:
            raise ValueError(f"{context}: expected either modelOutput or error")
        index += 1
    if ids != expected_ids:
        raise ValueError("output recordIds do not match input recordIds")


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


def _validate_tool_choice_tool_use(
    *,
    submit_rows: list[dict[str, object]],
    output_rows: list[dict[str, object]],
    settings: HarnessSettings,
) -> None:
    submit_index: dict[str, dict[str, object]] = {}
    for submit_entry in submit_rows:
        request_id_obj = submit_entry.get("request_id")
        if isinstance(request_id_obj, str):
            submit_index[request_id_obj] = submit_entry
    required_seen = False
    any_seen = False
    none_seen = False
    for output_row in output_rows:
        record_id_obj = output_row.get("recordId")
        if not isinstance(record_id_obj, str):
            continue
        context = f"recordId={record_id_obj}"
        submit_row = submit_index.get(record_id_obj)
        if not isinstance(submit_row, dict):
            continue
        agent_id_obj = submit_row.get("agent_id")
        step_id_obj = submit_row.get("step_id")
        if (
            not isinstance(agent_id_obj, str)
            or not isinstance(step_id_obj, str)
            or agent_id_obj != settings.tool_choice_agent_id
        ):
            continue
        model_output_obj = output_row.get("modelOutput")
        if not isinstance(model_output_obj, dict):
            continue
        has_tool_use = _has_tool_use(model_output_obj)
        if step_id_obj == settings.tool_choice_required_step_id:
            required_seen = True
            if not has_tool_use:
                raise ValueError(
                    f"tool-choice validation failed {context} step_id={step_id_obj}: expected tool call"
                )
        if step_id_obj == settings.tool_choice_any_step_id:
            any_seen = True
            if not has_tool_use:
                raise ValueError(
                    f"tool-choice validation failed {context} step_id={step_id_obj}: expected tool call"
                )
        if step_id_obj == settings.tool_choice_none_step_id:
            none_seen = True
            if has_tool_use:
                raise ValueError(
                    f"tool-choice validation failed {context} step_id={step_id_obj}: expected no tool call"
                )
    if not required_seen:
        raise ValueError(
            f"tool-choice validation failed: missing output row for step_id={settings.tool_choice_required_step_id}"
        )
    if not any_seen:
        raise ValueError(
            f"tool-choice validation failed: missing output row for step_id={settings.tool_choice_any_step_id}"
        )
    if not none_seen:
        raise ValueError(
            f"tool-choice validation failed: missing output row for step_id={settings.tool_choice_none_step_id}"
        )


def _validate_submit_rows(
    rows: list[dict[str, object]], expected_count: int, settings: HarnessSettings
) -> tuple[set[str], str]:
    if len(rows) <= expected_count:
        raise ValueError(
            "submit requests must include follow-up chain request rows beyond the initial batch"
        )
    ids: set[str] = set()
    root_chain_id = ""
    chain_follow_up_found = False
    file_manager_follow_up_found = False
    index = 0
    while index < len(rows):
        row = rows[index]
        context = f"submit row index={index}"
        request_id = _require_str(row, "request_id", context=context)
        if request_id in ids:
            raise ValueError(f"{context}: duplicate request_id={request_id}")
        ids.add(request_id)
        agent_id = _require_str(row, "agent_id", context=context)
        step_id = row.get("step_id")
        if index == 1:
            if agent_id != settings.chain_agent_id:
                raise ValueError(
                    f"{context}: expected second submit row agent_id={settings.chain_agent_id}, got {agent_id}"
                )
            if step_id != settings.chain_first_step_id:
                raise ValueError(
                    f"{context}: expected second submit row step_id={settings.chain_first_step_id}, got {step_id}"
                )
            root_chain_id = request_id
        if index == settings.file_manager_index:
            if agent_id != settings.file_manager_agent_id:
                raise ValueError(
                    f"{context}: expected submit row index={settings.file_manager_index} agent_id={settings.file_manager_agent_id}, got {agent_id}"
                )
            if row.get("tool_use_enabled") is not True:
                raise ValueError(
                    f"{context}: expected submit row index={settings.file_manager_index} tool_use_enabled=True"
                )
        if index == settings.tool_choice_index:
            if agent_id != settings.tool_choice_agent_id:
                raise ValueError(
                    f"{context}: expected submit row index={settings.tool_choice_index} agent_id={settings.tool_choice_agent_id}, got {agent_id}"
                )
            if step_id != settings.tool_choice_required_step_id:
                raise ValueError(
                    f"{context}: expected submit row index={settings.tool_choice_index} step_id={settings.tool_choice_required_step_id}, got {step_id}"
                )
        response_as_tool_obj = row.get("response_as_tool")
        response_as_tool = (
            response_as_tool_obj if isinstance(response_as_tool_obj, bool) else False
        )
        output_schema_obj = row.get("output_schema")
        has_structured_output = isinstance(output_schema_obj, str) and bool(
            output_schema_obj
        )
        tool_use_enabled = row.get("tool_use_enabled") is True
        model_obj = row.get("model")
        is_bedrock = False
        if isinstance(model_obj, dict):
            provider_obj = model_obj.get("provider")
            if provider_obj == "bedrock":
                is_bedrock = True
        if has_structured_output and tool_use_enabled:
            tools_obj = row.get("tools")
            tools_list = tools_obj if isinstance(tools_obj, list) else []
            has_final_result_tool = False
            for tool_obj in tools_list:
                if not isinstance(tool_obj, dict):
                    continue
                tool_name = tool_obj.get("name")
                if tool_name == "final_result":
                    has_final_result_tool = True
                    break
            has_response_format = row.get("response_format") is not None
            if response_as_tool:
                if not has_final_result_tool:
                    raise ValueError(
                        f"{context}: expected final_result tool when response_as_tool=true with structured output + tools"
                    )
                if has_response_format:
                    raise ValueError(
                        f"{context}: response_format must be omitted when response_as_tool=true with structured output + tools"
                    )
            elif is_bedrock:
                raise ValueError(
                    f"{context}: invalid config for Bedrock structured output + tools with response_as_tool=false"
                )
        if agent_id == settings.file_manager_agent_id and index >= expected_count:
            file_manager_follow_up_found = True
        if (
            agent_id == settings.chain_agent_id
            and step_id == settings.chain_second_step_id
            and index >= expected_count
        ):
            context_obj = _require_dict(row, "context", context=context)
            state_obj = _require_dict(
                context_obj, "state", context=f"{context} context"
            )
            steps_obj = _require_dict(state_obj, "steps", context=f"{context} state")
            random_name_obj = steps_obj.get(settings.chain_first_step_id)
            if not isinstance(random_name_obj, dict):
                raise ValueError(
                    f"{context}: expected dict state for step {settings.chain_first_step_id}"
                )
            random_name = random_name_obj.get("name")
            if not isinstance(random_name, str) or not random_name.strip():
                raise ValueError(
                    f"{context}: expected non-empty random name in state for {settings.chain_first_step_id}"
                )
            chain_follow_up_found = True
        index += 1
    if not root_chain_id:
        raise ValueError("unable to identify root chain request in submit rows")
    if not chain_follow_up_found:
        raise ValueError(
            "missing chain follow-up submit request with random-name state"
        )
    if not file_manager_follow_up_found:
        raise ValueError(
            "missing file manager follow-up submit request with tool results"
        )
    return ids, root_chain_id


def _validate_outcome_rows(
    rows: list[dict[str, object]],
    expected_count: int,
    chain_index: int,
    settings: HarnessSettings,
    record_ids_by_index: dict[int, str],
) -> OutcomeRowsResult:
    if len(rows) != expected_count:
        raise ValueError(
            f"outcome row count mismatch: expected={expected_count}, actual={len(rows)}"
        )
    result = OutcomeRowsResult()
    index = 0
    while index < len(rows):
        row = rows[index]
        record_id = record_ids_by_index.get(index)
        if record_id is None:
            raise ValueError(
                f"outcome row index={index}: missing mapped recordId from input rows"
            )
        context = f"outcome row index={index} recordId={record_id}"
        try:
            has_result = row.get("result") is not None
            has_next = row.get("next_request") is not None
            if not has_result:
                raise ValueError(f"{context}: missing result")
            if has_next:
                raise ValueError(
                    f"{context}: next_request is not expected in this harness flow"
                )
            if index == settings.file_manager_index:
                file_result = row["result"]
                if not isinstance(file_result, str) or not file_result.strip():
                    raise ValueError(
                        f"{context}: file manager result must be a non-empty string"
                    )
                lower = file_result.lower()
                if settings.file_manager_directory not in lower:
                    raise ValueError(
                        f"{context}: file manager result missing expected directory={settings.file_manager_directory!r}"
                    )
                if settings.file_manager_append_text not in lower:
                    raise ValueError(
                        f"{context}: file manager result missing append confirmation={settings.file_manager_append_text!r}"
                    )
                result.warnings_by_index[index] = []
                index += 1
                continue
            if index == settings.agent_memory_index:
                memory_result = row["result"]
                if not isinstance(memory_result, str) or not memory_result.strip():
                    raise ValueError(
                        f"{context}: agent memory result must be a non-empty string"
                    )
                expected_prefix = (
                    f"{settings.agent_memory_first_name} your random word is "
                )
                if not memory_result.lower().startswith(expected_prefix.lower()):
                    raise ValueError(
                        f"{context}: agent memory result does not match expected pattern "
                        f"{expected_prefix!r}, got {memory_result!r}"
                    )
                word = memory_result[len(expected_prefix) :].strip()
                if not word:
                    raise ValueError(
                        f"{context}: agent memory result missing random word after prefix"
                    )
                result.warnings_by_index[index] = []
                index += 1
                continue
            if record_id.startswith("code-rule-"):
                evaluation = _parse_evaluation(row["result"], context=context)
                if evaluation.status != "fail":
                    raise ValueError(
                        f"{context}: expected status=fail, got status={evaluation.status!r}"
                    )
                result.warnings_by_index[index] = []
                index += 1
                continue
            if record_id.startswith("harness-tool-choice-random-name-"):
                random_name_result = row["result"]
                try:
                    if isinstance(random_name_result, dict):
                        random_name = RandomName.model_validate(random_name_result)
                    elif isinstance(random_name_result, str):
                        try:
                            random_name = RandomName.model_validate_json(
                                random_name_result
                            )
                        except ValidationError:
                            start = random_name_result.find("{")
                            end = random_name_result.rfind("}")
                            if start >= 0 and end > start:
                                random_name = RandomName.model_validate_json(
                                    random_name_result[start : end + 1]
                                )
                            else:
                                raise ValueError(
                                    f"{context}: tool-choice result is not valid RandomName"
                                )
                    else:
                        raise ValueError(
                            f"{context}: unsupported tool-choice result type {type(random_name_result)}"
                        )
                except ValidationError as error:
                    raise ValueError(
                        f"{context}: tool-choice result is not valid RandomName"
                    ) from error
                if not random_name.name.strip():
                    raise ValueError(
                        f"{context}: tool-choice result missing non-empty name"
                    )
                result.warnings_by_index[index] = []
                index += 1
                continue
            tech_keywords = _parse_tech_keywords(row["result"], context=context)
            row_warnings = _validate_tech_keywords(tech_keywords, context=context)
            result.warnings_by_index[index] = row_warnings
            result.keywords_by_index[index] = tech_keywords
            if index == chain_index:
                if not tech_keywords.computer_languages:
                    raise ValueError(
                        f"{context}: chain final output missing computer_languages values"
                    )
        except ValueError as row_error:
            result.errors_by_index[index] = str(row_error)
        index += 1
    return result


def _validate_file_manager_file(settings: HarnessSettings) -> None:
    input_data = json.loads(settings.file_manager_input)
    if not isinstance(input_data, dict):
        raise ValueError("file_manager_input is not a JSON object")
    directory = input_data.get("directory")
    search_name = input_data.get("search_name")
    append_text = input_data.get("append_text")
    if (
        not isinstance(directory, str)
        or not isinstance(search_name, str)
        or not isinstance(append_text, str)
    ):
        raise ValueError(
            "file_manager_input missing required string fields: directory, search_name, append_text"
        )
    file_path = settings.safe_dir / directory / search_name
    if not file_path.exists():
        raise ValueError(f"file manager target file not found: {file_path}")
    content = file_path.read_text(encoding="utf-8")
    if append_text not in content:
        raise ValueError(
            f"file manager target file {search_name!r} missing expected append_text={append_text!r}"
        )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise ValueError(f"Missing expected output file: {path}")
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        item = json.loads(value)
        if not isinstance(item, dict):
            raise ValueError(f"Expected JSON object row in {path}, got: {type(item)}")
        rows.append(item)
    return rows


def verify(
    *,
    settings: HarnessSettings,
    input_jsonl: Path,
    submit_requests_jsonl: Path,
    output_jsonl: Path,
    outcomes_jsonl: Path,
    expected_count: int,
) -> VerificationResult:
    input_rows = _read_jsonl(input_jsonl)
    submit_rows = _read_jsonl(submit_requests_jsonl)
    output_rows = _read_jsonl(output_jsonl)
    outcome_rows = _read_jsonl(outcomes_jsonl)
    print(
        f"processing outcomes_jsonl={outcomes_jsonl},outcome_rows={len(outcome_rows)}"
    )

    input_errors: list[str] = []
    submit_errors: list[str] = []
    output_errors: list[str] = []
    outcome_errors: list[str] = []
    outcome_warnings: list[str] = []
    outcome_keywords: dict[int, object] = {}
    warnings_by_index: dict[int, list[str]] = {}
    input_ids: set[str] = set()
    submit_ids: set[str] = set()
    outcome_record_ids: dict[int, str] = {}

    index = 0
    while index < len(input_rows):
        row = input_rows[index]
        context = f"input row index={index}"
        record_id = _require_str(row, "recordId", context=context)
        outcome_record_ids[index] = record_id
        index += 1

    try:
        input_ids = _validate_input_rows(input_rows, expected_count)
    except ValueError as error:
        input_errors.append(str(error))

    try:
        submit_ids, _ = _validate_submit_rows(submit_rows, expected_count, settings)
    except ValueError as error:
        submit_errors.append(str(error))

    try:
        _validate_output_rows(output_rows, submit_ids)
        _validate_tool_choice_tool_use(
            submit_rows=submit_rows,
            output_rows=output_rows,
            settings=settings,
        )
        if len(output_rows) <= len(input_rows):
            raise ValueError(
                "output rows must include at least one additional row for chain follow-up execution"
            )
    except ValueError as error:
        output_errors.append(str(error))

    try:
        outcome_rows_result = _validate_outcome_rows(
            outcome_rows,
            expected_count,
            chain_index=1,
            settings=settings,
            record_ids_by_index=outcome_record_ids,
        )
        for warnings_list in outcome_rows_result.warnings_by_index.values():
            outcome_warnings.extend(warnings_list)
        outcome_keywords = outcome_rows_result.keywords_by_index
        warnings_by_index = outcome_rows_result.warnings_by_index
        if outcome_rows_result.errors_by_index:
            for idx, msg in outcome_rows_result.errors_by_index.items():
                outcome_errors.append(msg)
            outcome_errors.append(
                f"outcome row success rate: {outcome_rows_result.success_count}/{outcome_rows_result.total_count} ({outcome_rows_result.success_pct}%)"
            )
    except ValueError as error:
        outcome_errors.append(str(error))

    input_stage = StageResult(
        stage_name="input",
        file_path=str(input_jsonl),
        row_count=len(input_rows),
        errors=input_errors,
    )

    submit_stage = StageResult(
        stage_name="submit_requests",
        file_path=str(submit_requests_jsonl),
        row_count=len(submit_rows),
        errors=submit_errors,
    )

    output_stage = StageResult(
        stage_name="output",
        file_path=str(output_jsonl),
        row_count=len(output_rows),
        errors=output_errors,
    )

    outcome_stage = StageResult(
        stage_name="outcomes",
        file_path=str(outcomes_jsonl),
        row_count=len(outcome_rows),
        errors=outcome_errors,
        warnings=outcome_warnings,
    )

    overall_passed = (
        len(input_errors) == 0
        and len(submit_errors) == 0
        and len(output_errors) == 0
        and len(outcome_errors) == 0
    )

    result = VerificationResult(
        passed=overall_passed,
        input_stage=input_stage,
        submit_stage=submit_stage,
        output_stage=output_stage,
        outcome_stage=outcome_stage,
    )
    result.outcome_keywords = outcome_keywords
    result.outcome_warnings_by_index = warnings_by_index
    return result


def _format_results(result: VerificationResult) -> str:
    """Format verification results for console output."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("Bedrock Batch Test Harness Verification Results")
    lines.append("=" * 70)
    lines.append("")

    status_icon = "✓" if result.passed else "✗"
    status_text = "PASSED" if result.passed else "FAILED"
    lines.append(f"Status: {status_icon} {status_text}")
    lines.append("")

    lines.append("Validation Summary:")
    for stage in result.all_stages:
        stage_icon = "✓" if stage.passed else "✗"
        stage_status = "PASS" if stage.passed else "FAIL"
        warning_text = " ⚠" if stage.has_warnings else ""
        lines.append(
            f"  {stage.stage_name:20} {stage_icon} {stage_status:5} ({stage.row_count} rows){warning_text}"
        )

    if result.total_errors > 0:
        lines.append("")
        lines.append(f"Errors ({result.total_errors}):")
        for stage in result.all_stages:
            for error in stage.errors:
                lines.append(f"  [{stage.stage_name}] {error}")
        for error in result.errors:
            lines.append(f"  [harness] {error}")

    if result.total_warnings > 0:
        lines.append("")
        lines.append(f"Warnings ({result.total_warnings}):")
        for stage in result.all_stages:
            for warning in stage.warnings:
                lines.append(f"  [{stage.stage_name}] {warning}")
        for warning in result.warnings:
            lines.append(f"  [harness] {warning}")

    lines.append("")
    return "\n".join(lines)


def run(settings: HarnessSettings) -> None:
    results_dir = settings.harness_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    result = verify(
        settings=settings,
        input_jsonl=settings.input_jsonl,
        submit_requests_jsonl=settings.submit_requests_jsonl,
        output_jsonl=settings.output_jsonl,
        outcomes_jsonl=settings.outcomes_jsonl,
        expected_count=settings.count,
    )

    try:
        _validate_file_manager_file(settings)
    except ValueError as file_error:
        result.errors.append(str(file_error))
        result.passed = False

    formatted_output = _format_results(result)
    print(formatted_output, file=sys.stdout)
    logger.info(formatted_output)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracker = RequestTracker()

    try:
        submit_rows = _read_jsonl(settings.submit_requests_jsonl)
        input_rows = _read_jsonl(settings.input_jsonl)
        output_rows = _read_jsonl(settings.output_jsonl)
        outcome_rows = _read_jsonl(settings.outcomes_jsonl)

        input_status = "PASS" if result.input_stage.passed else "FAIL"
        output_status = "PASS" if result.output_stage.passed else "FAIL"
        outcome_status = "PASS" if result.outcome_stage.passed else "FAIL"

        input_errors_str = "; ".join(result.input_stage.errors)
        output_errors_str = "; ".join(result.output_stage.errors)
        outcome_errors_str = "; ".join(result.outcome_stage.errors)

        for idx, row in enumerate(submit_rows):
            request_id = row.get("request_id")
            agent_id = row.get("agent_id")
            step_id = row.get("step_id")
            if isinstance(request_id, str) and isinstance(agent_id, str):
                tracker.add_submit_request(
                    request_id,
                    agent_id,
                    step_id if isinstance(step_id, str) else None,
                )
                tracker.set_input_status(request_id, request_id, input_status)
                tracker.set_output_status(
                    request_id, output_status, input_errors_str or output_errors_str
                )

                outcome_warnings_str = ""
                outcome_error_str = ""
                tech_keywords_str = ""

                if idx in result.outcome_keywords:
                    keywords_obj = result.outcome_keywords[idx]
                    if hasattr(keywords_obj, "computer_languages"):
                        langs = getattr(keywords_obj, "computer_languages", [])
                        dbs = getattr(keywords_obj, "databases", [])
                        other = getattr(keywords_obj, "other", [])
                        tech_keywords_str = ", ".join(langs + dbs + other)

                if idx in result.outcome_warnings_by_index:
                    warnings_list = result.outcome_warnings_by_index[idx]
                    outcome_warnings_str = "; ".join(warnings_list)

                if outcome_errors_str:
                    outcome_error_str = outcome_errors_str

                tracker.set_outcome_status(
                    request_id, outcome_status, tech_keywords_str, outcome_warnings_str
                )
                if outcome_error_str:
                    if idx < len(outcome_rows):
                        tracker.requests[request_id].error_msg = outcome_error_str

        csv_files = export_csv_by_agent(tracker, results_dir, timestamp)
        summary_file = export_summary_csv(tracker, results_dir, timestamp)

        csv_output = (
            f"CSV exports generated: {len(csv_files)} agent files + 1 summary\n"
        )
        for csv_file in csv_files:
            csv_output += f"  📄 {csv_file.name}\n"
        csv_output += f"  📄 {summary_file.name}\n"
        print(csv_output, file=sys.stdout)
        logger.info(csv_output)
    except Exception as csv_error:
        logger.warning(f"CSV export failed: {csv_error}")

    if not result.passed:
        error_summary = f"Verification failed with {result.total_errors} error(s)"
        raise ValueError(error_summary)
