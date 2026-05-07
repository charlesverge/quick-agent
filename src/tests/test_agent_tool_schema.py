from __future__ import annotations

from pydantic import JsonValue

from quick_agent.agent_state import AgentState
from quick_agent.agent_tool_schema import (
  strip_agent_state_from_schema,
  takes_agent_state,
)


class SchemaNoState:
  pass


async def func_with_state(state: AgentState, urls: list[str]) -> list[str]:
  return urls


async def func_without_state(urls: list[str]) -> list[str]:
  return urls


def sync_func_with_state(state: AgentState, name: str) -> str:
  return name


class TestTakesAgentState:
  def test_detects_agent_state_first_param(self) -> None:
    assert takes_agent_state(func_with_state) is True

  def test_rejects_no_state_param(self) -> None:
    assert takes_agent_state(func_without_state) is False

  def test_detects_sync_function(self) -> None:
    assert takes_agent_state(sync_func_with_state) is True

  def test_rejects_non_agent_state_first_param(self) -> None:
    def other_ctx(ctx: SchemaNoState, x: int) -> int:
      return x
    assert takes_agent_state(other_ctx) is False


class TestStripAgentStateFromSchema:
  def test_strips_state_property_and_defs(self) -> None:
    schema: dict[str, JsonValue] = {
      "$defs": {
        "AgentState": {
          "properties": {"memory": {"additionalProperties": True, "type": "object"}},
          "required": ["memory"],
          "title": "AgentState",
          "type": "object",
        }
      },
      "additionalProperties": False,
      "properties": {
        "state": {"$ref": "#/$defs/AgentState"},
        "urls": {"items": {"type": "string"}, "type": "array"},
      },
      "required": ["state", "urls"],
      "type": "object",
    }
    result = strip_agent_state_from_schema(schema)
    properties = result["properties"]
    assert isinstance(properties, dict)
    assert "state" not in properties
    assert "$defs" not in result
    assert result["required"] == ["urls"]

  def test_preserves_other_defs(self) -> None:
    schema: dict[str, JsonValue] = {
      "$defs": {
        "AgentState": {"type": "object"},
        "OtherType": {"type": "string"},
      },
      "properties": {
        "state": {"$ref": "#/$defs/AgentState"},
        "data": {"$ref": "#/$defs/OtherType"},
      },
      "required": ["state", "data"],
      "type": "object",
    }
    result = strip_agent_state_from_schema(schema)
    properties = result["properties"]
    assert isinstance(properties, dict)
    assert "state" not in properties
    defs = result["$defs"]
    assert isinstance(defs, dict)
    assert "OtherType" in defs
    assert "AgentState" not in defs

  def test_does_not_mutate_original(self) -> None:
    schema: dict[str, JsonValue] = {
      "$defs": {"AgentState": {"type": "object"}},
      "properties": {
        "state": {"$ref": "#/$defs/AgentState"},
        "urls": {"items": {"type": "string"}, "type": "array"},
      },
      "required": ["state", "urls"],
      "type": "object",
    }
    strip_agent_state_from_schema(schema)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert "state" in properties
    defs = schema["$defs"]
    assert isinstance(defs, dict)
    assert "AgentState" in defs

  def test_no_properties_returns_schema(self) -> None:
    schema: dict[str, JsonValue] = {"type": "object"}
    result = strip_agent_state_from_schema(schema)
    assert result == {"type": "object"}
