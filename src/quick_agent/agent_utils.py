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
