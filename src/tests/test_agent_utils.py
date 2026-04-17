from __future__ import annotations

from quick_agent.agent_utils import (
    _normalize_anthropic,
    _normalize_converse,
    _normalize_openai,
    normalize_tool_calls,
)


def test_normalize_anthropic_basic() -> None:
    tc: dict[str, object] = {
        "type": "tool_use",
        "id": "id-1",
        "name": "my_tool",
        "input": {"key": "val"},
    }
    result = _normalize_anthropic(tc)
    assert result == {"id": "id-1", "name": "my_tool", "arguments": {"key": "val"}}


def test_normalize_anthropic_wrong_type() -> None:
    tc: dict[str, object] = {
        "type": "text",
        "id": "id-1",
        "name": "my_tool",
        "input": {},
    }
    assert _normalize_anthropic(tc) is None


def test_normalize_anthropic_none_input() -> None:
    tc: dict[str, object] = {
        "type": "tool_use",
        "id": "id-1",
        "name": "my_tool",
        "input": None,
    }
    result = _normalize_anthropic(tc)
    assert result is not None
    assert result["arguments"] is None


def test_normalize_converse_basic() -> None:
    tc: dict[str, object] = {
        "toolUse": {"toolUseId": "id-2", "name": "list_files", "input": {"path": "/"}}
    }
    result = _normalize_converse(tc)
    assert result == {"id": "id-2", "name": "list_files", "arguments": {"path": "/"}}


def test_normalize_converse_missing_key() -> None:
    tc: dict[str, object] = {"type": "tool_use", "id": "id-1", "name": "my_tool"}
    assert _normalize_converse(tc) is None


def test_normalize_openai_basic() -> None:
    tc: dict[str, object] = {
        "id": "call-1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
    }
    result = _normalize_openai(tc)
    assert result == {
        "id": "call-1",
        "name": "get_weather",
        "arguments": {"city": "NYC"},
    }


def test_normalize_openai_invalid_json_args() -> None:
    tc: dict[str, object] = {
        "id": "call-2",
        "type": "function",
        "function": {"name": "do_thing", "arguments": "not-json"},
    }
    result = _normalize_openai(tc)
    assert result is not None
    assert result["arguments"] == "not-json"


def test_normalize_openai_missing_function() -> None:
    tc: dict[str, object] = {"id": "call-3", "type": "function"}
    assert _normalize_openai(tc) is None


def test_normalize_tool_calls_mixed_formats() -> None:
    raw: list[dict[str, object]] = [
        {"type": "tool_use", "id": "a1", "name": "tool_a", "input": {"x": 1}},
        {"toolUse": {"toolUseId": "b2", "name": "tool_b", "input": {"y": 2}}},
        {
            "id": "c3",
            "type": "function",
            "function": {"name": "tool_c", "arguments": '{"z": 3}'},
        },
    ]
    result = normalize_tool_calls(raw)
    assert len(result) == 3
    assert result[0] == {"id": "a1", "name": "tool_a", "arguments": {"x": 1}}
    assert result[1] == {"id": "b2", "name": "tool_b", "arguments": {"y": 2}}
    assert result[2] == {"id": "c3", "name": "tool_c", "arguments": {"z": 3}}


def test_normalize_tool_calls_skips_unrecognized() -> None:
    raw: list[dict[str, object]] = [
        {"unknown_key": "value"},
        {"type": "tool_use", "id": "a1", "name": "tool_a", "input": {}},
    ]
    result = normalize_tool_calls(raw)
    assert len(result) == 1
    assert result[0]["name"] == "tool_a"


def test_normalize_tool_calls_empty() -> None:
    assert normalize_tool_calls([]) == []


def test_normalize_tool_calls_final_result_parses_json_arguments() -> None:
    raw: list[dict[str, object]] = [
        {
            "id": "c3",
            "name": "final_result",
            "arguments": '{"x": 3}',
        }
    ]
    result = normalize_tool_calls(raw)
    assert len(result) == 1
    assert result[0]["name"] == "final_result"
    assert result[0]["arguments"] == {"x": 3}
