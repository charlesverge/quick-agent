from __future__ import annotations

import json
from typing import Type

from pydantic import BaseModel

from quick_agent.json_utils import extract_first_json_object, json_compatible_value
from quick_agent.types import AgentResult


def normalize_usage_metrics(usage: object) -> dict[str, object]:
    usage_dict: dict[str, object] = {}
    if isinstance(usage, dict):
        for key, value in usage.items():
            usage_dict[str(key)] = json_compatible_value(value)
        return usage_dict
    if isinstance(usage, BaseModel):
        payload = usage.model_dump(exclude_none=True)
        if isinstance(payload, dict):
            for key, value in payload.items():
                usage_dict[str(key)] = json_compatible_value(value)
        return usage_dict
    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        payload = model_dump(exclude_none=True)
        if isinstance(payload, dict):
            for key, value in payload.items():
                usage_dict[str(key)] = json_compatible_value(value)
            return usage_dict
    return usage_dict


def parse_structured_result(
    raw_output: object, schema_cls: Type[BaseModel]
) -> BaseModel:
    if isinstance(raw_output, BaseModel):
        if isinstance(raw_output, schema_cls):
            return raw_output
        payload = raw_output.model_dump(mode="json")
        return schema_cls.model_validate(payload)
    if isinstance(raw_output, str):
        try:
            return schema_cls.model_validate_json(raw_output)
        except (json.JSONDecodeError, ValueError):
            cleaned_raw = extract_first_json_object(raw_output)
            return schema_cls.model_validate_json(cleaned_raw)
    return schema_cls.model_validate(raw_output)


def _as_agent_result(value: object) -> AgentResult:
    if isinstance(value, BaseModel):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, item in value.items():
            result[str(key)] = item
        return result
    if isinstance(value, list):
        items: list[AgentResult] = []
        index = 0
        while index < len(value):
            items.append(_as_agent_result(value[index]))
            index += 1
        return items
    raise ValueError(f"Unsupported completed batch output type: {type(value)}")


def extract_finish_reason(response: object | None) -> str | None:
    if response is None:
        return None
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    finish_reason = getattr(first_choice, "finish_reason", None)
    if isinstance(finish_reason, str) and finish_reason:
        return finish_reason
    return None


def agent_results_to_str(last_step_output: AgentResult) -> str:
    if isinstance(last_step_output, BaseModel):
        return last_step_output.model_dump_json(indent=2)
    elif isinstance(last_step_output, (dict, list)):
        return json.dumps(last_step_output, indent=2)
    return str(last_step_output)


def _normalize_anthropic(tc: dict[str, object]) -> dict[str, object] | None:
    if tc.get("type") != "tool_use":
        return None
    id_val = tc.get("id")
    name_val = tc.get("name")
    input_val = tc.get("input")
    return {
        "id": id_val
        if isinstance(id_val, (str, int, float, bool)) or id_val is None
        else str(id_val),
        "name": name_val
        if isinstance(name_val, (str, int, float, bool)) or name_val is None
        else str(name_val),
        "arguments": input_val
        if isinstance(input_val, (dict, list, str, int, float, bool))
        or input_val is None
        else str(input_val),
    }


def _normalize_converse(tc: dict[str, object]) -> dict[str, object] | None:
    tool_use = tc.get("toolUse")
    if not isinstance(tool_use, dict):
        return None
    tu_id = tool_use.get("toolUseId")
    tu_name = tool_use.get("name")
    tu_input = tool_use.get("input")
    return {
        "id": tu_id
        if isinstance(tu_id, (str, int, float, bool)) or tu_id is None
        else str(tu_id),
        "name": tu_name
        if isinstance(tu_name, (str, int, float, bool)) or tu_name is None
        else str(tu_name),
        "arguments": tu_input
        if isinstance(tu_input, (dict, list, str, int, float, bool)) or tu_input is None
        else str(tu_input),
    }


def _normalize_openai(tc: dict[str, object]) -> dict[str, object] | None:
    func = tc.get("function")
    tc_id = tc.get("id")
    if isinstance(func, dict):
        name_val = func.get("name")
        args_val = func.get("arguments")
    else:
        name_val = tc.get("name")
        args_val = tc.get("arguments")
    if not isinstance(name_val, (str, int, float, bool)) and name_val is not None:
        return None
    if name_val is None:
        return None
    parsed_args: object = args_val
    if isinstance(args_val, str):
        try:
            parsed_args = json.loads(args_val)
        except json.JSONDecodeError:
            parsed_args = args_val
    return {
        "id": tc_id
        if isinstance(tc_id, (str, int, float, bool)) or tc_id is None
        else str(tc_id),
        "name": name_val
        if isinstance(name_val, (str, int, float, bool)) or name_val is None
        else str(name_val),
        "arguments": parsed_args,
    }


def _normalize_final_result(tc: dict[str, object]) -> dict[str, object]:
    if tc.get("name") != "final_result":
        return tc
    args = tc.get("arguments")
    if isinstance(args, str):
        try:
            tc["arguments"] = json.loads(args)
        except json.JSONDecodeError:
            tc["arguments"] = args
    return tc


def normalize_tool_calls(raw: list[dict[str, object]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for tc in raw:
        normalized = (
            _normalize_anthropic(tc) or _normalize_converse(tc) or _normalize_openai(tc)
        )
        if normalized is not None:
            result.append(_normalize_final_result(normalized))
    return result
