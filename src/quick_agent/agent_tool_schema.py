from __future__ import annotations

import inspect
import typing
from typing import Any, Callable

from pydantic import JsonValue

from quick_agent.agent_state import AgentState


def takes_agent_state(func: Callable[..., Any]) -> bool:
  sig = inspect.signature(func)
  first_param = next(iter(sig.parameters), None)
  if first_param is None:
    return False
  try:
    hints = typing.get_type_hints(func)
  except Exception:
    return False
  return hints.get(first_param) is AgentState


def strip_agent_state_from_schema(
  schema: dict[str, JsonValue],
) -> dict[str, JsonValue]:
  schema = dict(schema)
  properties = schema.get("properties")
  if not isinstance(properties, dict):
    return schema
  first_key = next(iter(properties), None)
  if first_key is None:
    return schema
  properties = dict(properties)
  del properties[first_key]
  schema["properties"] = properties
  required = schema.get("required")
  if isinstance(required, list) and first_key in required:
    schema["required"] = [r for r in required if r != first_key]
  defs = schema.get("$defs")
  if isinstance(defs, dict):
    filtered = {k: v for k, v in defs.items() if k != "AgentState"}
    if filtered:
      schema["$defs"] = filtered
    else:
      del schema["$defs"]
  return schema
