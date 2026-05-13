from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

from quick_agent.models.model_spec import ModelSettings, ModelSpec
from quick_agent.recorder import ExecutionLogEntry, Recorder


@dataclass
class DummyConfig:
    agent_id: str
    model_spec: ModelSpec
    tool_ids: list[str]


@dataclass
class DummyContext:
    effective_base_url: str


@dataclass
class DummyExecutor:
    config: DummyConfig
    context: DummyContext


def _make_recorder(*, enable_llm_request_logging: bool = False, llm_log_path: Path | str | None = None, http_log_max_entries: int = 3) -> Recorder:
    executor = DummyExecutor(
        config=DummyConfig(
            agent_id="agent",
            model_spec=ModelSpec(base_url="https://api.openai.com/v1", model_name="m"),
            tool_ids=["t1"],
        ),
        context=DummyContext(effective_base_url="https://api.openai.com/v1"),
    )
    return Recorder(
        executor=executor,
        http_log_max_entries=http_log_max_entries,
        enable_llm_request_logging=enable_llm_request_logging,
        llm_log_path=llm_log_path,
    )


def test_write_llm_request_log_writes_when_enabled(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "out.log"
    recorder = _make_recorder(enable_llm_request_logging=True, llm_log_path=path)
    recorder._write_llm_request_log({"key": "value"})
    assert path.exists()
    contents = path.read_text(encoding="utf-8")
    assert "[LLM_REQUEST]" in contents
    assert '"key": "value"' in contents


def test_write_llm_request_log_no_op_when_disabled(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    recorder = _make_recorder(enable_llm_request_logging=False, llm_log_path=path)
    recorder._write_llm_request_log({"x": 1})
    assert not path.exists()


def test_write_llm_request_log_no_op_when_payload_none(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    recorder = _make_recorder(enable_llm_request_logging=True, llm_log_path=path)
    recorder._write_llm_request_log(None)
    assert not path.exists()


def test_write_llm_request_log_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dirs" / "out.log"
    recorder = _make_recorder(enable_llm_request_logging=True, llm_log_path=path)
    recorder._write_llm_request_log({"hello": "world"})
    assert path.exists()
    assert path.parent.exists()


def test_write_llm_request_log_appends_multiple_entries(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    recorder = _make_recorder(enable_llm_request_logging=True, llm_log_path=path)
    recorder._write_llm_request_log({"a": 1})
    recorder._write_llm_request_log({"b": 2})
    text = path.read_text(encoding="utf-8")
    assert text.count("[LLM_REQUEST]") == 2


def test_write_llm_request_log_handles_oserror_gracefully(tmp_path: Path) -> None:
    path = tmp_path / "dir"
    path.mkdir()
    recorder = _make_recorder(enable_llm_request_logging=True, llm_log_path=path)
    recorder._write_llm_request_log({"x": 1})
    assert path.exists()


@pytest.mark.anyio
async def test_record_http_request_captures_method_url_headers_body() -> None:
    recorder = _make_recorder()
    request = httpx.Request("POST", "http://x/y", headers={"H": "V"}, content=b'{"a":1}')
    await recorder._record_http_request(request)
    assert len(recorder.http_request_log) == 1
    entry = recorder.http_request_log[0]
    assert entry["method"] == "POST"
    assert entry["url"] == "http://x/y"
    assert entry["headers"].get("H", entry["headers"].get("h")) == "V"
    assert entry["body"] == '{"a":1}'


@pytest.mark.anyio
async def test_record_http_response_includes_status_and_headers() -> None:
    recorder = _make_recorder()
    request = httpx.Request("POST", "http://x/y", headers={"H": "V"}, content=b'{"a":1}')
    response = httpx.Response(200, request=request, headers={"R": "V"}, content=b'hello')
    await recorder._record_http_response(response)
    assert len(recorder.http_response_log) == 1
    response_entry = recorder.http_response_log[0]
    assert response_entry["status_code"] == 200
    assert response_entry["headers"].get("R", response_entry["headers"].get("r")) == "V"
    assert response_entry["body"] == "hello"


def test_http_log_respects_max_entries() -> None:
    recorder = _make_recorder(http_log_max_entries=3)
    recorder._record_http_request_entry({"method": "GET"})
    recorder._record_http_request_entry({"method": "GET2"})
    recorder._record_http_request_entry({"method": "GET3"})
    recorder._record_http_request_entry({"method": "GET4"})
    assert len(recorder.http_request_log) == 3
    assert recorder.http_request_log[0]["method"] == "GET2"


def test_decode_http_bytes_handles_non_utf8() -> None:
    recorder = _make_recorder()
    result = recorder._decode_http_bytes(b"\xff\xfe invalid utf-8")
    assert "invalid utf-8" in result


def test_last_http_exchange_context_returns_latest_request_response_pair() -> None:
    recorder = _make_recorder()
    recorder.http_request_log.append({"method": "GET"})
    recorder.http_response_log.append({"status_code": 200})
    ctx = recorder._last_http_exchange_context()
    assert ctx["request"]["method"] == "GET"
    assert ctx["response"]["status_code"] == 200
    assert ctx["request_source"] == "quick_agent_http_traffic_log"


def test_last_http_exchange_context_falls_back_to_traffic_entries_when_logs_empty() -> None:
    recorder = _make_recorder()
    recorder._record_http_traffic_entry({"event": "request", "request": {"method": "GET"}})
    ctx = recorder._last_http_exchange_context()
    assert ctx["request"]["method"] == "GET"


def test_last_http_exchange_context_returns_empty_when_no_traffic() -> None:
    recorder = _make_recorder()
    assert recorder._last_http_exchange_context() == {}


def test_to_curl_builds_full_curl_command_from_request_context() -> None:
    request_context = {
        "request": {
            "method": "POST",
            "url": "https://api.x/v1/chat",
            "headers": {"Content-Type": "application/json", "Authorization": "Bearer abc"},
            "body": '{"k":"v"}',
        }
    }
    entry = ExecutionLogEntry(request_context=request_context, call_site="test")
    curl = entry.to_curl()
    assert curl.startswith("curl -X POST")
    assert "-H 'Content-Type: application/json'" in curl
    assert "-H 'Authorization: Bearer abc'" in curl
    assert "--data-raw '{\"k\":\"v\"}'" in curl


def test_to_curl_reconstructs_from_base_url_when_request_missing() -> None:
    request_context = {
        "base_url": "http://example.com/v1",
        "model_name": "m",
        "user_prompt": "hello",
        "system_prompt": "sys",
        "instructions": "inst",
    }
    entry = ExecutionLogEntry(request_context=request_context, call_site="test")
    curl = entry.to_curl()
    assert "http://example.com/v1/chat/completions" in curl
    assert "system" in curl
    assert "hello" in curl


def test_to_curl_handles_system_prompt_list() -> None:
    request_context = {
        "base_url": "http://example.com/v1",
        "model_name": "m",
        "user_prompt": "hello",
        "system_prompt": ["sys1", "sys2"],
    }
    entry = ExecutionLogEntry(request_context=request_context, call_site="test")
    curl = entry.to_curl()
    assert "sys1\\nsys2" in curl or "sys1\n sys2" in curl


def test_to_curl_appends_instructions_to_system_message() -> None:
    request_context = {
        "base_url": "http://example.com/v1",
        "model_name": "m",
        "user_prompt": "hello",
        "system_prompt": "sys",
        "instructions": "inst",
    }
    entry = ExecutionLogEntry(request_context=request_context, call_site="test")
    curl = entry.to_curl()
    assert "sys\\ninst" in curl


def test_to_curl_returns_minimal_curl_when_no_context_available() -> None:
    entry = ExecutionLogEntry(request_context={}, call_site="test")
    assert entry.to_curl() == "curl"


def test_to_curl_uses_chat_completions_url_when_base_url_has_no_endpoint() -> None:
    request_context = {
        "base_url": "http://example.com/v1",
        "model_name": "m",
        "user_prompt": "hello",
    }
    entry = ExecutionLogEntry(request_context=request_context, call_site="test")
    assert "/chat/completions" in entry.to_curl()


def test_to_curl_preserves_chat_completions_url_when_already_present() -> None:
    request_context = {
        "base_url": "http://example.com/v1/chat/completions",
        "model_name": "m",
        "user_prompt": "hello",
    }
    entry = ExecutionLogEntry(request_context=request_context, call_site="test")
    assert "/chat/completions" in entry.to_curl()


def test_record_llm_request_appends_to_execution_log() -> None:
    recorder = _make_recorder()
    recorder._record_llm_request(
        call_site="test",
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="i",
        system_prompt="s",
        user_prompt="u",
        model_settings=ModelSettings(),
    )
    assert len(recorder.execution_log) == 1
    assert recorder.execution_log[0].call_site == "test"


def test_record_llm_request_respects_max_entries_truncation() -> None:
    recorder = _make_recorder(http_log_max_entries=2)
    for _ in range(5):
        recorder._record_llm_request(
            call_site="test",
            step_id="s1",
            step_kind="text",
            output_schema=None,
            instructions="i",
            system_prompt="s",
            user_prompt="u",
            model_settings=ModelSettings(),
        )
    assert len(recorder.execution_log) == 2


def test_record_llm_request_payload_includes_all_fields_when_logging_enabled(tmp_path: Path) -> None:
    path = tmp_path / "out.log"
    recorder = _make_recorder(enable_llm_request_logging=True, llm_log_path=path)
    recorder._record_llm_request(
        call_site="test",
        step_id="s1",
        step_kind="text",
        output_schema="schema",
        instructions="inst",
        system_prompt="sys",
        user_prompt="user",
        model_settings=ModelSettings(),
    )
    text = path.read_text(encoding="utf-8")
    assert "agent_id" in text
    assert "model_name" in text
    assert "step" in text
    assert "system_prompt" in text
