from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import openai
import pytest

from quick_agent.agent_config import AgentConfig
from quick_agent.agent_state import AgentState
from quick_agent.executor import AgentExecutor, ToolCallResult, _should_convert_null
from quick_agent.models.agent_spec import AgentSpec
from quick_agent.models.batch_request import (
    BatchAgentContext,
    BatchImportRequest,
    BatchMessage,
    BatchModelConfig,
    BatchSubmitRequest,
    BatchToolDefinition,
)
from quick_agent.models.handoff_spec import HandoffSpec
from quick_agent.models.loaded_agent_file import LoadedAgentFile
from quick_agent.models.model_spec import ModelSpec
from quick_agent.models.output_spec import OutputSpec
from quick_agent.models.run_input import RunInput


class FakeCreateNamespace:
    def __init__(self) -> None:
        self.call_kwargs: dict[str, object] | None = None
        self.response: object | None = None
        self.error: Exception | None = None

    async def create(self, **kwargs: object) -> object:
        self.call_kwargs = kwargs
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=FakeCreateNamespace())


@dataclass
class FakeToolFunctionSchema:
    json_schema: dict[str, object]


@dataclass
class FakeTool:
    name: str
    description: str | None
    function_schema: FakeToolFunctionSchema
    function: object


@dataclass
class FakeToolset:
    tools: dict[str, FakeTool]


def _make_loaded() -> LoadedAgentFile:
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="https://api.openai.com/v1", model_name="m"),
        chain=[],
        schemas={},
        output=OutputSpec(),
        handoff=HandoffSpec(),
    )
    return LoadedAgentFile.from_parts(
        spec=spec,
        instructions="",
        system_prompt="",
        step_prompts={},
    )


def _make_executor(*, client: object | None = None, toolset: object | None = None, tool_ids: list[str] | None = None, convert_null: bool | None = None, base_url: str = "https://api.openai.com/v1", batch_call: object | None = None) -> AgentExecutor:
    loaded = _make_loaded()
    config = AgentConfig(
        agent_id="agent",
        toolset=toolset,
        tool_ids=tool_ids or [],
        memory={},
        model_spec=ModelSpec(
            provider="openai-compatible",
            base_url=base_url,
            model_name="m",
            convert_null=convert_null,
        ),
        client=client,
        http_client=None,
        extra_headers=None,
        extra_body=None,
        record_http_traffic=False,
        run_input=RunInput(source_path="in.txt", kind="text", text="hi", data=None),
        loaded=loaded,
        extra_tools=None,
        recorder=None,
        state={},
        batch_call=batch_call,
    )
    return AgentExecutor(config)


def _make_batch_request(*, messages: list[BatchMessage], tools: list[BatchToolDefinition] | None = None, response_format: dict[str, object] | None = None, max_tool_calls: int = 3) -> BatchSubmitRequest:
    return BatchSubmitRequest(
        request_id="req-1",
        agent_id="agent",
        step_id=None,
        step_kind="text",
        output_schema=None,
        model=BatchModelConfig(provider="openai-compatible", base_url="https://api.openai.com/v1", model_name="m"),
        messages=messages,
        response_format=response_format,
        tool_choice=None,
        max_tool_calls=max_tool_calls,
        tool_ids=[],
        tools=tools,
        tool_use_enabled=False,
        response_as_tool=False,
        final_result_tool_enabled=False,
        bedrock_model_id=None,
        context=BatchAgentContext(),
    )


def _make_toolset_with_state_tool() -> FakeToolset:
    def tool(state: AgentState, x: int) -> str:
        return f"x={x}"

    schema = {"type": "object", "properties": {"state": {"type": "object"}, "x": {"type": "integer"}}, "required": ["state", "x"]}
    return FakeToolset(
        tools={
            "tool_with_state": FakeTool(
                name="tool_with_state",
                description="desc",
                function_schema=FakeToolFunctionSchema(json_schema=schema),
                function=tool,
            )
        }
    )


def _make_toolset_with_simple_tool() -> FakeToolset:
    def tool(x: int) -> str:
        return str(x)

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    return FakeToolset(
        tools={
            "simple_tool": FakeTool(
                name="simple_tool",
                description="desc",
                function_schema=FakeToolFunctionSchema(json_schema=schema),
                function=tool,
            )
        }
    )


def _make_api_status_error(monkeypatch: pytest.MonkeyPatch, body: object) -> Exception:
    class FakeAPIStatusError(Exception):
        pass

    monkeypatch.setattr(openai, "APIStatusError", FakeAPIStatusError)
    error = FakeAPIStatusError("fail")
    setattr(error, "body", body)
    return error


def test_should_convert_null_detects_ollama() -> None:
    assert _should_convert_null("http://localhost:11434/v1/ollama")
    assert not _should_convert_null("https://api.openai.com/v1")


@pytest.mark.anyio
async def test_local_batch_call_builds_system_user_messages() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    request = _make_batch_request(
        messages=[
            BatchMessage(role="system", content="sys"),
            BatchMessage(role="user", content="hi"),
        ]
    )
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        id="resp1",
        usage={},
    )
    await executor._local_batch_call(request)
    assert fake.chat.completions.call_kwargs is not None
    assert fake.chat.completions.call_kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


@pytest.mark.anyio
async def test_local_batch_call_assistant_with_tool_calls_preserves_content_none() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    request = _make_batch_request(
        messages=[
            BatchMessage(role="assistant", content=None, tool_calls=[{"id": "c1", "function": {"name": "foo", "arguments": "{}"}}]),
        ]
    )
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="foo", arguments="{}"))]))],
        id="resp2",
        usage={},
    )
    await executor._local_batch_call(request)
    assert fake.chat.completions.call_kwargs is not None
    message = fake.chat.completions.call_kwargs["messages"][0]
    assert message["role"] == "assistant"
    assert "content" not in message
    assert message["tool_calls"] == [{"id": "c1", "function": {"name": "foo", "arguments": "{}"}}]


@pytest.mark.anyio
async def test_local_batch_call_assistant_null_content_converted_for_ollama() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake, base_url="http://localhost:11434/v1/ollama")
    request = _make_batch_request(
        messages=[BatchMessage(role="assistant", content=None)],
    )
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok", tool_calls=None))],
        id="resp3",
        usage={},
    )
    await executor._local_batch_call(request)
    assert fake.chat.completions.call_kwargs is not None
    message = fake.chat.completions.call_kwargs["messages"][0]
    assert message["role"] == "assistant"
    assert message["content"] == ""


@pytest.mark.anyio
async def test_local_batch_call_tool_message_passes_tool_call_id() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    request = _make_batch_request(
        messages=[BatchMessage(role="tool", content="result text", tool_call_id="call-123")],
    )
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        id="resp4",
        usage={},
    )
    await executor._local_batch_call(request)
    assert fake.chat.completions.call_kwargs is not None
    message = fake.chat.completions.call_kwargs["messages"][0]
    assert message == {
        "role": "tool",
        "content": "result text",
        "tool_call_id": "call-123",
    }


@pytest.mark.anyio
async def test_local_batch_call_tool_message_with_null_content_becomes_empty_string() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    request = _make_batch_request(
        messages=[BatchMessage(role="tool", content=None, tool_call_id="call-1")],
    )
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        id="resp5",
        usage={},
    )
    await executor._local_batch_call(request)
    assert fake.chat.completions.call_kwargs is not None
    assert fake.chat.completions.call_kwargs["messages"][0]["content"] == ""


@pytest.mark.anyio
async def test_local_batch_call_includes_tools_from_batch_request() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    tools = [
        BatchToolDefinition(
            name="get_weather",
            description="Get weather",
            input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        )
    ]
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")], tools=tools)
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        id="resp6",
        usage={},
    )
    await executor._local_batch_call(request)
    assert fake.chat.completions.call_kwargs is not None
    tool_payload = fake.chat.completions.call_kwargs["tools"]
    assert isinstance(tool_payload, list)
    assert tool_payload[0]["type"] == "function"
    assert tool_payload[0]["function"]["name"] == "get_weather"


@pytest.mark.anyio
async def test_local_batch_call_falls_back_to_toolset_when_no_explicit_tools() -> None:
    fake = FakeAsyncOpenAI()
    toolset = _make_toolset_with_simple_tool()
    executor = _make_executor(client=fake, toolset=toolset, tool_ids=["simple_tool"])
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")], tools=None)
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        id="resp7",
        usage={},
    )
    await executor._local_batch_call(request)
    assert fake.chat.completions.call_kwargs is not None
    tool_payload = fake.chat.completions.call_kwargs["tools"]
    assert tool_payload[0]["function"]["name"] == "simple_tool"


@pytest.mark.anyio
async def test_local_batch_call_strips_agent_state_from_toolset_schemas() -> None:
    fake = FakeAsyncOpenAI()
    toolset = _make_toolset_with_state_tool()
    executor = _make_executor(client=fake, toolset=toolset, tool_ids=["tool_with_state"])
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")], tools=None)
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        id="resp8",
        usage={},
    )
    await executor._local_batch_call(request)
    assert fake.chat.completions.call_kwargs is not None
    tool_payload = fake.chat.completions.call_kwargs["tools"]
    assert "state" not in tool_payload[0]["function"]["parameters"]
    assert "properties" in tool_payload[0]["function"]["parameters"]
    assert tool_payload[0]["function"]["parameters"]["properties"]["x"] == {"type": "integer"}


@pytest.mark.anyio
async def test_local_batch_call_omits_tools_kwarg_when_empty() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake, tool_ids=[])
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")], tools=None)
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        id="resp9",
        usage={},
    )
    await executor._local_batch_call(request)
    assert fake.chat.completions.call_kwargs is not None
    assert fake.chat.completions.call_kwargs["tools"] is openai.omit


@pytest.mark.anyio
async def test_local_batch_call_returns_completed_for_content_response() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello world"))],
        id="resp10",
        usage={},
    )
    result = await executor._local_batch_call(request)
    assert result.payload == {"state": "completed", "output": "hello world"}


@pytest.mark.anyio
async def test_local_batch_call_returns_tool_use_for_tool_call_response() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    request = _make_batch_request(messages=[BatchMessage(role="assistant", content=None)])
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="my_tool", arguments='{"x":1}'))]))],
        id="resp11",
        usage={},
    )
    result = await executor._local_batch_call(request)
    assert result.payload["state"] == "tool_use"
    assert result.payload["tool_calls"] == [{"id": "c1", "name": "my_tool", "arguments": '{"x":1}'}]
    assert result.payload["submit_request"] == request.model_dump(mode="json")


@pytest.mark.anyio
async def test_local_batch_call_raises_for_empty_response() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=None, refusal=None))],
        id="resp12",
        usage={},
    )
    with pytest.raises(ValueError, match="Model returned an empty response."):
        await executor._local_batch_call(request)


@pytest.mark.anyio
async def test_local_batch_call_raises_for_refusal_response() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=None, refusal="I cannot help with that"))],
        id="resp13",
        usage={},
    )
    with pytest.raises(ValueError, match="I cannot help with that"):
        await executor._local_batch_call(request)


@pytest.mark.anyio
async def test_local_batch_call_raises_when_tool_rounds_exhausted() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    request = _make_batch_request(messages=[BatchMessage(role="assistant", tool_calls=[{"id": "c1", "function": {"name": "foo", "arguments": "{}"}}])], max_tool_calls=1)
    fake.chat.completions.response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c1", function=SimpleNamespace(name="foo", arguments="{}"))]))],
        id="resp14",
        usage={},
    )
    with pytest.raises(ValueError, match="Max tool call rounds reached"):
        await executor._local_batch_call(request)


@pytest.mark.anyio
async def test_local_batch_call_maps_tools_not_supported_apistatuserror_to_qa_exception() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    error = openai.APIStatusError(
        "fail",
        response=httpx.Response(400, request=httpx.Request("GET", "http://example.com")),
        body={"message": "X does not support tools"},
    )
    fake.chat.completions.error = error
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    with pytest.raises(Exception) as excinfo:
        await executor._local_batch_call(request)
    assert excinfo.type.__name__ == "QuickAgentToolsNotSupportedException"
    assert excinfo.value.args[0] == "X does not support tools"


@pytest.mark.anyio
async def test_local_batch_call_maps_chat_not_supported_apistatuserror_to_qa_exception() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    error = openai.APIStatusError(
        "fail",
        response=httpx.Response(400, request=httpx.Request("GET", "http://example.com")),
        body={"message": "X does not support chat"},
    )
    fake.chat.completions.error = error
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    with pytest.raises(Exception) as excinfo:
        await executor._local_batch_call(request)
    assert excinfo.type.__name__ == "QuickAgentChatNotSupportedException"


@pytest.mark.anyio
async def test_local_batch_call_reraises_unmapped_apistatuserror() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    error = openai.APIStatusError(
        "fail",
        response=httpx.Response(400, request=httpx.Request("GET", "http://example.com")),
        body={"message": "some other error"},
    )
    fake.chat.completions.error = error
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    with pytest.raises(openai.APIStatusError):
        await executor._local_batch_call(request)


@pytest.mark.anyio
async def test_local_batch_call_maps_error_from_string_body() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    error = openai.APIStatusError(
        "fail",
        response=httpx.Response(400, request=httpx.Request("GET", "http://example.com")),
        body="X does not support tools",
    )
    fake.chat.completions.error = error
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    with pytest.raises(Exception) as excinfo:
        await executor._local_batch_call(request)
    assert excinfo.type.__name__ == "QuickAgentToolsNotSupportedException"


@pytest.mark.anyio
async def test_local_batch_call_captures_metrics_with_finish_reason() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="hello", refusal=None), finish_reason="stop")],
        id="resp-15",
        usage={"prompt_tokens": 10},
        model="m-final",
        created=123,
        system_fingerprint="fingerprint",
    )
    fake.chat.completions.response = response
    request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    await executor._local_batch_call(request)
    assert executor.last_run_metrics is not None
    assert executor.last_run_metrics["provider"] == "openai-compatible"
    assert executor.last_run_metrics["model"] == "m-final"
    assert executor.last_run_metrics["completion_id"] == "resp-15"
    assert executor.last_run_metrics["finish_reason"] == "stop"


@pytest.mark.anyio
async def test_execute_batch_request_completes_immediately_when_no_tool_calls() -> None:
    executor = _make_executor()
    async def fake_call(batch_request: BatchSubmitRequest) -> BatchImportRequest:
        return BatchImportRequest(request_id="req", payload={"state": "completed", "output": "final answer"})

    executor._call_batch_handler = fake_call
    result = await executor._execute_batch_request(batch_request=_make_batch_request(messages=[BatchMessage(role="user", content="hi")]), schema_cls=None)
    assert result == "final answer"


@pytest.mark.anyio
async def test_execute_batch_request_loops_on_tool_use_then_completes() -> None:
    executor = _make_executor()
    first_request = _make_batch_request(messages=[BatchMessage(role="assistant", content=None)])
    responses = [
        BatchImportRequest(
            request_id="req",
            payload={
                "state": "tool_use",
                "tool_calls": [{"id": "c1", "name": "t", "arguments": {}}],
                "submit_request": first_request.model_dump(mode="json"),
            },
        ),
        BatchImportRequest(request_id="req", payload={"state": "completed", "output": "done"}),
    ]

    async def fake_call(batch_request: BatchSubmitRequest) -> BatchImportRequest:
        return responses.pop(0)

    executor._call_batch_handler = fake_call

    async def fake_execute_tool_calls(tool_calls: list[dict[str, object]]) -> list[ToolCallResult]:
        return [ToolCallResult(id="c1", name="t", content="result")]

    executor._execute_tool_calls = fake_execute_tool_calls
    result = await executor._execute_batch_request(batch_request=first_request, schema_cls=None)
    assert result == "done"


@pytest.mark.anyio
async def test_execute_batch_request_follows_submit_next_state() -> None:
    executor = _make_executor()
    next_request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    responses = [
        BatchImportRequest(request_id="req", payload={"state": "submit_next", "next_request": next_request.model_dump(mode="json")} ),
        BatchImportRequest(request_id="req", payload={"state": "completed", "output": "final"}),
    ]

    async def fake_call(batch_request: BatchSubmitRequest) -> BatchImportRequest:
        return responses.pop(0)

    executor._call_batch_handler = fake_call
    result = await executor._execute_batch_request(batch_request=_make_batch_request(messages=[BatchMessage(role="user", content="hi")]), schema_cls=None)
    assert result == "final"


@pytest.mark.anyio
async def test_execute_batch_request_raises_on_max_tool_calls() -> None:
    executor = _make_executor()
    batch_request = _make_batch_request(messages=[BatchMessage(role="assistant", tool_calls=[{"id": "c1", "function": {"name": "foo", "arguments": "{}"}}])], max_tool_calls=1)
    async def fake_call(batch_request: BatchSubmitRequest) -> BatchImportRequest:
        return BatchImportRequest(
            request_id=batch_request.request_id,
            payload={
                "state": "tool_use",
                "tool_calls": [{"id": "c1", "name": "t", "arguments": {}}],
                "submit_request": batch_request.model_dump(mode="json"),
            },
        )
    executor._call_batch_handler = fake_call
    with pytest.raises(ValueError, match="Max tool call rounds reached"):
        await executor._execute_batch_request(batch_request=batch_request, schema_cls=None)


@pytest.mark.anyio
async def test_execute_batch_request_raises_when_tool_use_missing_pending_submit_request() -> None:
    executor = _make_executor()
    async def fake_call(batch_request: BatchSubmitRequest) -> BatchImportRequest:
        return BatchImportRequest(request_id="req", payload={"state": "tool_use", "tool_calls": [{"id": "c1", "name": "t", "arguments": {}}]})

    executor._call_batch_handler = fake_call
    with pytest.raises(ValueError, match="tool_use outcome is missing pending_submit_request"):
        await executor._execute_batch_request(batch_request=_make_batch_request(messages=[BatchMessage(role="user", content="hi")]), schema_cls=None)


@pytest.mark.anyio
async def test_execute_batch_request_raises_when_result_none_with_text_schema() -> None:
    executor = _make_executor()
    async def fake_call(batch_request: BatchSubmitRequest) -> BatchImportRequest:
        return BatchImportRequest(request_id="req", payload={"state": "completed", "output": None})
    executor._call_batch_handler = fake_call
    with pytest.raises(ValueError, match="Unsupported completed batch output type: <class 'NoneType'>"):
        await executor._execute_batch_request(
            batch_request=_make_batch_request(messages=[BatchMessage(role="user", content="hi")]),
            schema_cls=None,
        )


@pytest.mark.anyio
async def test_execute_batch_request_raises_when_text_step_returns_non_string() -> None:
    executor = _make_executor()
    async def fake_call(batch_request: BatchSubmitRequest) -> BatchImportRequest:
        return BatchImportRequest(request_id="req", payload={"state": "completed", "output": {"x": 1}})
    executor._call_batch_handler = fake_call
    with pytest.raises(ValueError, match="Text step expected a string output"):
        await executor._execute_batch_request(batch_request=_make_batch_request(messages=[BatchMessage(role="user", content="hi")]), schema_cls=None)


@pytest.mark.anyio
async def test_execute_batch_request_parses_result_for_structured_schema() -> None:
    from pydantic import BaseModel

    class ExampleSchema(BaseModel):
        x: int

    executor = _make_executor()

    async def fake_call(batch_request: BatchSubmitRequest) -> BatchImportRequest:
        return BatchImportRequest(request_id="req", payload={"state": "completed", "output": '{"x": 42}'})

    executor._call_batch_handler = fake_call
    result = await executor._execute_batch_request(
        batch_request=_make_batch_request(messages=[BatchMessage(role="user", content="hi")]),
        schema_cls=ExampleSchema,
    )
    assert isinstance(result, ExampleSchema)
    assert result.x == 42


@pytest.mark.anyio
async def test_call_batch_handler_falls_back_to_local_call_when_handler_none() -> None:
    fake = FakeAsyncOpenAI()
    executor = _make_executor(client=fake)
    called: list[bool] = []

    async def fake_local(batch_request: BatchSubmitRequest) -> BatchImportRequest:
        called.append(True)
        return BatchImportRequest(request_id="req", payload={"state": "completed", "output": "ok"})

    executor._local_batch_call = fake_local
    result = await executor._call_batch_handler(_make_batch_request(messages=[BatchMessage(role="user", content="hi")]))
    assert result.payload["output"] == "ok"
    assert called == [True]


@pytest.mark.anyio
async def test_call_batch_handler_handles_sync_handler() -> None:
    def handler(batch_request: BatchSubmitRequest) -> BatchImportRequest:
        return BatchImportRequest(request_id=batch_request.request_id, payload={"state": "completed", "output": "ok"})

    executor = _make_executor(batch_call=handler)
    result = await executor._call_batch_handler(_make_batch_request(messages=[BatchMessage(role="user", content="hi")]))
    assert result.payload["output"] == "ok"


@pytest.mark.anyio
async def test_call_batch_handler_handles_async_handler() -> None:
    async def handler(batch_request: BatchSubmitRequest) -> BatchImportRequest:
        return BatchImportRequest(request_id=batch_request.request_id, payload={"state": "completed", "output": "ok"})

    executor = _make_executor(batch_call=handler)
    result = await executor._call_batch_handler(_make_batch_request(messages=[BatchMessage(role="user", content="hi")]))
    assert result.payload["output"] == "ok"


@pytest.mark.anyio
async def test_call_batch_handler_validates_dict_return_into_batchimportrequest() -> None:
    def handler(batch_request: BatchSubmitRequest) -> dict[str, object]:
        return {"request_id": batch_request.request_id, "payload": {"state": "completed", "output": "ok"}}

    executor = _make_executor(batch_call=handler)
    result = await executor._call_batch_handler(_make_batch_request(messages=[BatchMessage(role="user", content="hi")]))
    assert isinstance(result, BatchImportRequest)
    assert result.payload["output"] == "ok"


@pytest.mark.anyio
async def test_call_batch_handler_rejects_non_callable_handler() -> None:
    executor = _make_executor(batch_call="not a function")
    with pytest.raises(ValueError, match="config.batch_call must be callable"):
        await executor._call_batch_handler(_make_batch_request(messages=[BatchMessage(role="user", content="hi")]))


def test_import_outcome_error_state_raises_value_error_for_unmapped_message() -> None:
    executor = _make_executor()
    payload = {"state": "error", "message": "random model failure"}
    with pytest.raises(ValueError, match="random model failure"):
        executor.import_outcome(batch_import=BatchImportRequest(request_id="req", payload=payload))


def test_import_outcome_error_state_missing_message_raises() -> None:
    executor = _make_executor()
    payload = {"state": "error"}
    with pytest.raises(ValueError, match="Error batch payload is missing string field 'message'."):
        executor.import_outcome(batch_import=BatchImportRequest(request_id="req", payload=payload))


def test_import_outcome_completed_state_missing_output_raises() -> None:
    executor = _make_executor()
    payload = {"state": "completed"}
    with pytest.raises(ValueError, match="Completed batch payload is missing 'output'."):
        executor.import_outcome(batch_import=BatchImportRequest(request_id="req", payload=payload))


def test_import_outcome_submit_next_missing_next_request_raises() -> None:
    executor = _make_executor()
    payload = {"state": "submit_next"}
    with pytest.raises(ValueError, match="submit_next batch payload is missing object field 'next_request'."):
        executor.import_outcome(batch_import=BatchImportRequest(request_id="req", payload=payload))


def test_import_outcome_unsupported_state_raises() -> None:
    executor = _make_executor()
    payload = {"state": "unknown_state"}
    with pytest.raises(ValueError, match="Unsupported batch import state: unknown_state"):
        executor.import_outcome(batch_import=BatchImportRequest(request_id="req", payload=payload))


def test_import_outcome_final_result_with_missing_arguments_raises() -> None:
    executor = _make_executor()
    submit_request_payload = _make_batch_request(messages=[BatchMessage(role="user", content="hi")]).model_dump(mode="json")
    submit_request_payload["final_result_tool_enabled"] = True
    payload = {
        "state": "tool_use",
        "tool_calls": [{"id": "final-1", "name": "final_result"}],
        "submit_request": submit_request_payload,
    }
    outcome = executor.import_outcome(batch_import=BatchImportRequest(request_id="req", payload=payload))
    assert outcome.tool_calls is not None
    assert outcome.tool_calls[0]["name"] == "final_result"


def test_import_outcome_state_field_must_be_string() -> None:
    executor = _make_executor()
    payload = {"state": 123}
    with pytest.raises(ValueError, match="Batch import payload is missing string field 'state'."):
        executor.import_outcome(batch_import=BatchImportRequest(request_id="req", payload=payload))


def test_build_next_request_serializes_dict_args_to_json_string() -> None:
    executor = _make_executor()
    submit_request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    tool_calls = [{"id": "c1", "name": "t", "arguments": {"x": 1, "y": 2}}]
    executed = [ToolCallResult(id="c1", name="t", content="result")]
    next_request = executor._build_next_request_with_tool_results(
        tool_calls=tool_calls, executed=executed, submit_request=submit_request
    )
    assert next_request.messages[-2].tool_calls[0]["function"]["arguments"] == json.dumps({"x": 1, "y": 2})


def test_build_next_request_handles_non_dict_non_string_args_as_empty_object() -> None:
    executor = _make_executor()
    submit_request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    tool_calls = [{"id": "c1", "name": "t", "arguments": None}]
    executed = [ToolCallResult(id="c1", name="t", content="result")]
    next_request = executor._build_next_request_with_tool_results(
        tool_calls=tool_calls, executed=executed, submit_request=submit_request
    )
    assert next_request.messages[-2].tool_calls[0]["function"]["arguments"] == "{}"


def test_build_next_request_uses_string_args_as_is() -> None:
    executor = _make_executor()
    submit_request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    tool_calls = [{"id": "c1", "name": "t", "arguments": '{"already":"json"}'}]
    executed = [ToolCallResult(id="c1", name="t", content="result")]
    next_request = executor._build_next_request_with_tool_results(
        tool_calls=tool_calls, executed=executed, submit_request=submit_request
    )
    assert next_request.messages[-2].tool_calls[0]["function"]["arguments"] == '{"already":"json"}'


def test_build_next_request_uses_tool_error_as_content_when_present() -> None:
    executor = _make_executor()
    submit_request = _make_batch_request(messages=[BatchMessage(role="user", content="hi")])
    tool_calls = [{"id": "c1", "name": "t", "arguments": {}}]
    executed = [ToolCallResult(id="c1", name="t", content=None, error="permission denied")]
    next_request = executor._build_next_request_with_tool_results(
        tool_calls=tool_calls, executed=executed, submit_request=submit_request
    )
    assert next_request.messages[-1].content == "permission denied"
