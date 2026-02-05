import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.toolsets import FunctionToolset

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import TextInput
from quick_agent.models import AgentSpec
from quick_agent.models import ChainStepSpec
from quick_agent.models import LoadedAgentFile
from quick_agent.models import ModelSpec
from quick_agent.models.output_spec import OutputSpec
from quick_agent.quick_agent import QuickAgent
from quick_agent import quick_agent as qa_module


def dummy_tool() -> str:
    return "ok"


class HttpxRequestRecorder:
    def __init__(self, response_json: dict[str, Any]) -> None:
        self.response_json = response_json
        self.requests: list[httpx.Request] = []
        self.last_json: dict[str, Any] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.last_json = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=self.response_json)


class StaticRegistry(AgentRegistry):
    def __init__(self, loaded: LoadedAgentFile) -> None:
        super().__init__([])
        self._loaded = loaded

    def get(self, agent_id: str) -> LoadedAgentFile:
        return self._loaded


class BuildModelStub:
    def __init__(self, model: OpenAIChatModel) -> None:
        self.model = model

    def __call__(self, _: ModelSpec) -> OpenAIChatModel:
        return self.model


def _chat_completion_response(model_name: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 123,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _messages_by_role(messages: list[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [message for message in messages if message.get("role") == role]


@pytest.mark.anyio
async def test_single_shot_without_tools_omits_tools_in_httpx_post(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response_json = _chat_completion_response("gpt-5")
    recorder = HttpxRequestRecorder(response_json)
    transport = httpx.MockTransport(recorder)

    async with httpx.AsyncClient(transport=transport, base_url="https://example.test/v1") as client:
        provider = OpenAIProvider(base_url="https://example.test/v1", api_key="test", http_client=client)
        model = OpenAIChatModel("gpt-5", provider=provider)
        monkeypatch.setattr(qa_module, "build_model", BuildModelStub(model))

        step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
        spec = AgentSpec(
            name="test",
            model=ModelSpec(base_url="https://example.test/v1", model_name="gpt-5"),
            chain=[step],
            tools=[],
            output=OutputSpec(file=None),
        )
        loaded = LoadedAgentFile.from_parts(
            spec=spec,
            instructions="system",
            system_prompt="",
            step_prompts={"step:one": "say hi"},
        )

        registry = StaticRegistry(loaded)
        tools = AgentTools([])
        permissions = DirectoryPermissions(tmp_path)

        agent = QuickAgent(
            registry=registry,
            tools=tools,
            directory_permissions=permissions,
            agent_id="agent-1",
            input_data=TextInput("hello"),
            extra_tools=None,
            write_output=False,
        )

        result = await agent.run()

    assert result == "ok"
    assert len(recorder.requests) == 1
    assert recorder.last_json is not None
    assert recorder.last_json.get("tools") is None
    assert recorder.last_json.get("tool_choice") is None


@pytest.mark.anyio
async def test_single_shot_with_tools_includes_tools_in_httpx_post(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response_json = _chat_completion_response("gpt-5")
    recorder = HttpxRequestRecorder(response_json)
    transport = httpx.MockTransport(recorder)

    async with httpx.AsyncClient(transport=transport, base_url="https://example.test/v1") as client:
        provider = OpenAIProvider(base_url="https://example.test/v1", api_key="test", http_client=client)
        model = OpenAIChatModel("gpt-5", provider=provider)
        monkeypatch.setattr(qa_module, "build_model", BuildModelStub(model))

        step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
        spec = AgentSpec(
            name="test",
            model=ModelSpec(base_url="https://example.test/v1", model_name="gpt-5"),
            chain=[step],
            tools=["dummy.tool"],
            output=OutputSpec(file=None),
        )
        loaded = LoadedAgentFile.from_parts(
            spec=spec,
            instructions="system",
            system_prompt="",
            step_prompts={"step:one": "say hi"},
        )

        registry = StaticRegistry(loaded)
        tools = AgentTools([])
        toolset = FunctionToolset[Any]()
        toolset.add_function(func=dummy_tool, name="dummy_tool", description="dummy tool")
        monkeypatch.setattr(tools, "build_toolset", lambda *_: toolset)
        permissions = DirectoryPermissions(tmp_path)

        agent = QuickAgent(
            registry=registry,
            tools=tools,
            directory_permissions=permissions,
            agent_id="agent-1",
            input_data=TextInput("hello"),
            extra_tools=None,
            write_output=False,
        )

        result = await agent.run()

    assert result == "ok"
    assert len(recorder.requests) == 1
    assert recorder.last_json is not None
    tools_json = recorder.last_json.get("tools")
    assert isinstance(tools_json, list)
    assert tools_json
    assert tools_json[0]["function"]["name"] == "dummy_tool"


@pytest.mark.anyio
async def test_single_shot_no_steps_system_prompt_only_includes_system_prompt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response_json = _chat_completion_response("gpt-5")
    recorder = HttpxRequestRecorder(response_json)
    transport = httpx.MockTransport(recorder)

    async with httpx.AsyncClient(transport=transport, base_url="https://example.test/v1") as client:
        provider = OpenAIProvider(base_url="https://example.test/v1", api_key="test", http_client=client)
        model = OpenAIChatModel("gpt-5", provider=provider)
        monkeypatch.setattr(qa_module, "build_model", BuildModelStub(model))

        spec = AgentSpec(
            name="test",
            model=ModelSpec(base_url="https://example.test/v1", model_name="gpt-5"),
            chain=[],
            tools=[],
            output=OutputSpec(file=None),
        )
        loaded = LoadedAgentFile.from_parts(
            spec=spec,
            instructions="",
            system_prompt="You are concise.",
            step_prompts={},
        )

        registry = StaticRegistry(loaded)
        tools = AgentTools([])
        permissions = DirectoryPermissions(tmp_path)

        agent = QuickAgent(
            registry=registry,
            tools=tools,
            directory_permissions=permissions,
            agent_id="agent-1",
            input_data=TextInput("hello"),
            extra_tools=None,
            write_output=False,
        )

        result = await agent.run()

    assert result == "ok"
    assert recorder.last_json is not None
    messages = recorder.last_json.get("messages")
    assert isinstance(messages, list)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are concise."
    assert messages[-1]["role"] == "user"
    assert "# Task Input" not in messages[-1]["content"]
    assert "## Input Content" not in messages[-1]["content"]
    assert "## Chain State (YAML)" not in messages[-1]["content"]
    assert "## Step Instructions" not in messages[-1]["content"]


@pytest.mark.anyio
async def test_single_shot_no_steps_instructions_only_includes_instructions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    response_json = _chat_completion_response("gpt-5")
    recorder = HttpxRequestRecorder(response_json)
    transport = httpx.MockTransport(recorder)

    async with httpx.AsyncClient(transport=transport, base_url="https://example.test/v1") as client:
        provider = OpenAIProvider(base_url="https://example.test/v1", api_key="test", http_client=client)
        model = OpenAIChatModel("gpt-5", provider=provider)
        monkeypatch.setattr(qa_module, "build_model", BuildModelStub(model))

        spec = AgentSpec(
            name="test",
            model=ModelSpec(base_url="https://example.test/v1", model_name="gpt-5"),
            chain=[],
            tools=[],
            output=OutputSpec(file=None),
        )
        loaded = LoadedAgentFile.from_parts(
            spec=spec,
            instructions="Use the tool.",
            system_prompt="",
            step_prompts={},
        )

        registry = StaticRegistry(loaded)
        tools = AgentTools([])
        permissions = DirectoryPermissions(tmp_path)

        agent = QuickAgent(
            registry=registry,
            tools=tools,
            directory_permissions=permissions,
            agent_id="agent-1",
            input_data=TextInput("hello"),
            extra_tools=None,
            write_output=False,
        )

        result = await agent.run()

    assert result == "ok"
    assert recorder.last_json is not None
    messages = recorder.last_json.get("messages")
    assert isinstance(messages, list)
    system_messages = _messages_by_role(messages, "system")
    system_contents = [message.get("content") for message in system_messages]
    assert "Use the tool." in system_contents
    assert messages[-1]["role"] == "user"
    assert "# Task Input" not in messages[-1]["content"]
    assert "## Input Content" not in messages[-1]["content"]
    assert "## Chain State (YAML)" not in messages[-1]["content"]
    assert "## Step Instructions" not in messages[-1]["content"]
