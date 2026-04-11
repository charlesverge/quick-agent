"""Verification stage for bedrock batch test harness."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from schemas.tech_keywords import TechKeywords
from settings import HarnessSettings

REQUIRED_LANGUAGES = {"python", "typescript"}
REQUIRED_DATABASES = {"mongodb", "sql"}
CHAIN_AGENT_ID = "harness-language-chain-extractor"
CHAIN_FIRST_STEP_ID = "generate_random_name"
CHAIN_SECOND_STEP_ID = "tech_keyword_extraction"


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


def _validate_tech_keywords(keywords: TechKeywords, *, context: str) -> None:
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
        raise ValueError(
            f"{context}: missing required computer_languages value node.js"
        )

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
        raise ValueError(f"{context}: missing required extracted term react")


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
) -> None:
    if len(rows) != expected_count:
        raise ValueError(
            f"outcome row count mismatch: expected={expected_count}, actual={len(rows)}"
        )
    index = 0
    while index < len(rows):
        row = rows[index]
        context = f"outcome row index={index}"
        has_result = row.get("final_result") is not None
        has_next = row.get("next_submit_request") is not None
        if has_result == has_next:
            raise ValueError(
                f"{context}: must include exactly one of final_result or next_submit_request"
            )
        if has_next:
            raise ValueError(
                f"{context}: next_submit_request is not expected in this harness flow"
            )
        tech_keywords = _parse_tech_keywords(row["final_result"], context=context)
        _validate_tech_keywords(tech_keywords, context=context)
        if index == chain_index:
            if not tech_keywords.computer_languages:
                raise ValueError(
                    f"{context}: chain final output missing computer_languages values"
                )
        index += 1


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
) -> None:
    input_rows = _read_jsonl(input_jsonl)
    submit_rows = _read_jsonl(submit_requests_jsonl)
    output_rows = _read_jsonl(output_jsonl)
    outcome_rows = _read_jsonl(outcomes_jsonl)
    input_ids = _validate_input_rows(input_rows, expected_count)
    submit_ids, _ = _validate_submit_rows(submit_rows, expected_count)
    _validate_output_rows(output_rows, submit_ids)
    if len(output_rows) <= len(input_rows):
        raise ValueError(
            "output rows must include at least one additional row for chain follow-up execution"
        )
    _validate_outcome_rows(outcome_rows, expected_count, chain_index=1)


def run(settings: HarnessSettings) -> None:
    verify(
        input_jsonl=settings.input_jsonl,
        submit_requests_jsonl=settings.submit_requests_jsonl,
        output_jsonl=settings.output_jsonl,
        outcomes_jsonl=settings.outcomes_jsonl,
        expected_count=settings.count,
    )
