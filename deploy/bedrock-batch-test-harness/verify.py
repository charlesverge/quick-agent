"""Verification stage for bedrock batch test harness."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from models import StageResult, VerificationResult
from pydantic import ValidationError
from schemas.tech_keywords import TechKeywords
from settings import HarnessSettings
from verify_csv import RequestTracker, export_csv_by_agent, export_summary_csv

logger = logging.getLogger(__name__)

REQUIRED_LANGUAGES = {"python", "typescript"}
REQUIRED_DATABASES = {"mongodb", "sql"}
CHAIN_AGENT_ID = "harness-language-chain-extractor"
CHAIN_FIRST_STEP_ID = "generate-random-name"
CHAIN_SECOND_STEP_ID = "tech-keyword-extraction"


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
            f"{context}: missing required computer_languages values {sorted(missing_languages)}"
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
        context = f"output row index={index}"
        record_id = _require_str(row, "recordId", context=context)
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


def _validate_submit_rows(
    rows: list[dict[str, object]], expected_count: int
) -> tuple[set[str], str]:
    if len(rows) <= expected_count:
        raise ValueError(
            "submit requests must include follow-up chain request rows beyond the initial batch"
        )
    ids: set[str] = set()
    root_chain_id = ""
    chain_follow_up_found = False
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
            if agent_id != CHAIN_AGENT_ID:
                raise ValueError(
                    f"{context}: expected second submit row agent_id={CHAIN_AGENT_ID}, got {agent_id}"
                )
            if step_id != CHAIN_FIRST_STEP_ID:
                raise ValueError(
                    f"{context}: expected second submit row step_id={CHAIN_FIRST_STEP_ID}, got {step_id}"
                )
            root_chain_id = request_id
        if (
            agent_id == CHAIN_AGENT_ID
            and step_id == CHAIN_SECOND_STEP_ID
            and index >= expected_count
        ):
            context_obj = _require_dict(row, "context", context=context)
            state_obj = _require_dict(
                context_obj, "state", context=f"{context} context"
            )
            steps_obj = _require_dict(state_obj, "steps", context=f"{context} state")
            random_name_obj = steps_obj.get(CHAIN_FIRST_STEP_ID)
            if not isinstance(random_name_obj, dict):
                raise ValueError(
                    f"{context}: expected dict state for step {CHAIN_FIRST_STEP_ID}"
                )
            random_name = random_name_obj.get("name")
            if not isinstance(random_name, str) or not random_name.strip():
                raise ValueError(
                    f"{context}: expected non-empty random name in state for {CHAIN_FIRST_STEP_ID}"
                )
            chain_follow_up_found = True
        index += 1
    if not root_chain_id:
        raise ValueError("unable to identify root chain request in submit rows")
    if not chain_follow_up_found:
        raise ValueError(
            "missing chain follow-up submit request with random-name state"
        )
    return ids, root_chain_id


def _validate_outcome_rows(
    rows: list[dict[str, object]], expected_count: int, chain_index: int
) -> tuple[dict[int, list[str]], dict[int, TechKeywords]]:
    if len(rows) != expected_count:
        raise ValueError(
            f"outcome row count mismatch: expected={expected_count}, actual={len(rows)}"
        )
    warnings_by_index: dict[int, list[str]] = {}
    keywords_by_index: dict[int, TechKeywords] = {}
    index = 0
    while index < len(rows):
        row = rows[index]
        context = f"outcome row index={index}"
        has_result = row.get("result") is not None
        has_next = row.get("next_request") is not None
        if not has_result:
            raise ValueError(
                f"{context}: missing result"
            )
        if has_next:
            raise ValueError(
                f"{context}: next_request is not expected in this harness flow"
            )
        tech_keywords = _parse_tech_keywords(row["result"], context=context)
        row_warnings = _validate_tech_keywords(tech_keywords, context=context)
        warnings_by_index[index] = row_warnings
        keywords_by_index[index] = tech_keywords
        if index == chain_index:
            if not tech_keywords.computer_languages:
                raise ValueError(
                    f"{context}: chain final output missing computer_languages values"
                )
        index += 1
    return warnings_by_index, keywords_by_index


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

    input_errors: list[str] = []
    submit_errors: list[str] = []
    output_errors: list[str] = []
    outcome_errors: list[str] = []
    outcome_warnings: list[str] = []
    outcome_keywords: dict[int, object] = {}
    warnings_by_index: dict[int, list[str]] = {}
    input_ids: set[str] = set()
    submit_ids: set[str] = set()

    try:
      input_ids = _validate_input_rows(input_rows, expected_count)
    except ValueError as error:
      input_errors.append(str(error))

    try:
      submit_ids, _ = _validate_submit_rows(submit_rows, expected_count)
    except ValueError as error:
      submit_errors.append(str(error))

    try:
      _validate_output_rows(output_rows, submit_ids)
      if len(output_rows) <= len(input_rows):
        raise ValueError(
          "output rows must include at least one additional row for chain follow-up execution"
        )
    except ValueError as error:
      output_errors.append(str(error))

    try:
      warnings_by_index, keywords_by_index = _validate_outcome_rows(
        outcome_rows, expected_count, chain_index=1
      )
      for warnings_list in warnings_by_index.values():
        outcome_warnings.extend(warnings_list)
      outcome_keywords = keywords_by_index
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
        input_jsonl=settings.input_jsonl,
        submit_requests_jsonl=settings.submit_requests_jsonl,
        output_jsonl=settings.output_jsonl,
        outcomes_jsonl=settings.outcomes_jsonl,
        expected_count=settings.count,
    )

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
          tracker.add_submit_request(request_id, agent_id, step_id)
          tracker.set_input_status(request_id, request_id, input_status)
          tracker.set_output_status(request_id, output_status, input_errors_str or output_errors_str)

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

      csv_output = f"CSV exports generated: {len(csv_files)} agent files + 1 summary\n"
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
