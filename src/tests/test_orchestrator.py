import asyncio
import json
import sys
import types
from pathlib import Path
from typing import Any, Literal, cast

import httpx
import pytest
from pydantic import BaseModel, ValidationError
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import Tool
from pydantic_ai.toolsets import FunctionToolset

from quick_agent import agent_tools as tools_module
from quick_agent import input_adaptors as input_adaptors_module
from quick_agent import quick_agent as qa_module
from quick_agent import single_shot as single_shot_module
from quick_agent.agent_call_tool import AgentCallTool
from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.exceptions import (
    QuickAgentChatNotSupportedException,
    QuickAgentToolsNotSupportedException,
)
from quick_agent.models import AgentSpec, ChainStepSpec, LoadedAgentFile, ModelSpec
from quick_agent.models.batch_request import (
    BatchImportOutcome,
    BatchImportRequest,
    BatchMessage,
    BatchModelConfig,
    BatchSubmitRequest,
    BatchToolDefinition,
)
from quick_agent.models.content_processing_spec import (
    ChunkProcessingSpec,
    ContentProcessingSpec,
    SampleSpec,
)
from quick_agent.models.handoff_spec import HandoffSpec
from quick_agent.models.output_spec import OutputSpec
from quick_agent.models.run_input import RunInput
from quick_agent.orchestrator import Orchestrator
from quick_agent.output import write_output
from quick_agent.prompting import make_user_prompt
from quick_agent.quick_agent import (
    QuickAgent,
    ToolCallResult,
    build_model,
    resolve_schema,
)
from quick_agent.recorder import ExecutionLogEntry


class DummyProvider:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        http_client: httpx.AsyncClient | None = None,
        openai_client: object | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key


class DummyModel:
    def __init__(self, model_name: str, provider: DummyProvider) -> None:
        self.model_name = model_name
        self.provider = provider


class DummyOpenAIProvider:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url


class DummyOpenAIModel:
    def __init__(self, base_url: str) -> None:
        self.provider = DummyOpenAIProvider(base_url)


class RecordingToolset(FunctionToolset[Any]):
    def __init__(self) -> None:
        super().__init__()
        self.add_calls: list[tuple[Any, str, str]] = []

    def add_function(self, *args: Any, **kwargs: Any) -> Tool[Any]:
        func = kwargs.get("func")
        name = kwargs.get("name")
        description = kwargs.get("description")
        if func is not None and name is not None and description is not None:
            self.add_calls.append((func, name, description))
        return super().add_function(*args, **kwargs)


class FakeAgentResult:
    def __init__(
        self, output: Any, *, usage: object = None, response: object = None
    ) -> None:
        self.output = output
        self.response = response
        self._usage = usage

    def usage(self) -> object:
        return self._usage


class FakeAgent:
    next_output: Any = ""
    next_error: Exception | None = None
    next_usage: object = {}
    next_response: object = None
    last_init: dict[str, Any] | None = None
    last_prompt: str | None = None
    last_deps: Any = None

    def __init__(
        self,
        model: Any,
        instructions: str | None,
        system_prompt: str | list[str],
        toolsets: list[Any],
        output_type: Any,
        model_settings: Any | None = None,
    ) -> None:
        FakeAgent.last_init = {
            "model": model,
            "instructions": instructions,
            "system_prompt": system_prompt,
            "toolsets": toolsets,
            "output_type": output_type,
            "model_settings": model_settings,
        }

    async def run(self, user_prompt: str, *, deps: Any = None) -> FakeAgentResult:
        FakeAgent.last_prompt = user_prompt
        FakeAgent.last_deps = deps
        if FakeAgent.next_error is not None:
            error = FakeAgent.next_error
            FakeAgent.next_error = None
            raise error
        usage = FakeAgent.next_usage
        response = FakeAgent.next_response
        FakeAgent.next_usage = {}
        FakeAgent.next_response = None
        return FakeAgentResult(FakeAgent.next_output, usage=usage, response=response)


class OpenAIResponseStub:
    def __init__(
        self,
        content: str,
        *,
        usage: object = None,
        completion_id: str = "cmpl-123",
        model: str = "gpt-4.1-mini",
        created: int = 123,
        system_fingerprint: str = "fp-1",
        finish_reason: str = "stop",
    ) -> None:
        self.id = completion_id
        self.model = model
        self.created = created
        self.system_fingerprint = system_fingerprint
        self.usage = usage
        self.choices = [
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content, refusal=None),
                finish_reason=finish_reason,
            )
        ]


class LoadToolsRecorder:
    def __init__(self, toolset: Any) -> None:
        self.toolset = toolset
        self.calls: list[tuple[list[Path], list[str], DirectoryPermissions]] = []

    def __call__(
        self,
        tool_roots: list[Path],
        tool_ids: list[str],
        permissions: DirectoryPermissions,
    ) -> Any:
        self.calls.append((tool_roots, tool_ids, permissions))
        return self.toolset


class SyncCallRecorder:
    def __init__(self, return_value: Any = None) -> None:
        self.return_value = return_value
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return self.return_value


class AsyncCallRecorder:
    def __init__(self, return_value: Any = None) -> None:
        self.return_value = return_value
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append({"args": args, "kwargs": kwargs})
        return self.return_value


class FakeRegistry(AgentRegistry):
    def __init__(self, loaded: LoadedAgentFile) -> None:
        super().__init__(agent_roots=[])
        self.loaded = loaded
        self.calls: list[str] = []

    def get(self, agent_id: str) -> LoadedAgentFile:
        self.calls.append(agent_id)
        return self.loaded


class RecordingQuickAgent(QuickAgent):
    def __init__(self, outputs: list[tuple[Any, Any]]) -> None:
        self.outputs = outputs
        self.calls: list[str] = []
        self.index = 0

    async def _run_step(self, **kwargs: Any) -> Any:
        step = kwargs.get("step")
        if step is not None:
            self.calls.append(step.id)
        output = self.outputs[self.index][0]
        self.index += 1
        return output


class HandoffQuickAgent(QuickAgent):
    def __init__(self) -> None:
        self.calls: list[tuple[str, input_adaptors_module.InputAdaptor | Path]] = []

    async def _run_nested_agent(
        self, agent_id: str, input_data: input_adaptors_module.InputAdaptor | Path
    ) -> str:
        self.calls.append((agent_id, input_data))
        return "ok"


class ExampleSchema(BaseModel):
    x: int


class OutputSchema(BaseModel):
    msg: str


class OtherSchema(BaseModel):
    msg: str


def _make_loaded_with_chain(
    chain: list[ChainStepSpec],
    *,
    schemas: dict[str, str] | None = None,
    output: OutputSpec | None = None,
    handoff: HandoffSpec | None = None,
) -> LoadedAgentFile:
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=chain,
        schemas=schemas or {},
        output=output or OutputSpec(file="out/result.json"),
        handoff=handoff or HandoffSpec(),
    )
    return LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )


def _permissions(tmp_path: Path | None = None) -> DirectoryPermissions:
    root = Path("safe") if tmp_path is None else tmp_path / "safe"
    return DirectoryPermissions(root)


def _make_quick_agent_for_test(
    *,
    loaded: LoadedAgentFile | None = None,
    run_input: RunInput | None = None,
    model: OpenAIChatModel | None = None,
    toolset: FunctionToolset[Any] | None = None,
    memory: dict[str, Any] | None = None,
    enable_llm_request_logging: bool = False,
    llm_log_path: Path | str | None = None,
) -> QuickAgent:
    if loaded is None:
        loaded = _make_loaded_with_chain(
            [ChainStepSpec(id="s1", kind="text", prompt_section="step:one")]
        )
    if run_input is None:
        run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)
    registry = FakeRegistry(loaded)
    tools = AgentTools([])
    agent = QuickAgent(
        registry=registry,
        tools=tools,
        directory_permissions=_permissions(),
        agent_id="a",
        input_data=input_adaptors_module.TextInput(run_input.text),
        extra_tools=None,
        model=loaded.spec.model,
        write_output=False,
        record_http_traffic=False,
        memory=memory,
        enable_llm_request_logging=enable_llm_request_logging,
        llm_log_path=llm_log_path,
    )
    agent.run_input = run_input
    agent.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
    if model is not None:
        agent.model = model
    if toolset is not None:
        agent.toolset = toolset
    return agent


def test_init_sets_registry_and_tool_roots(tmp_path: Path) -> None:
    orch = Orchestrator(
        [tmp_path], [tmp_path / "tools"], safe_dir=_permissions(tmp_path).root
    )

    assert isinstance(orch.registry, AgentRegistry)
    assert isinstance(orch.tools, AgentTools)
    assert orch.tools._tool_roots == [tmp_path / "tools"]


def test_build_model_uses_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "abc")
    monkeypatch.setattr(qa_module, "OpenAIProvider", DummyProvider)
    monkeypatch.setattr(qa_module, "OpenAIChatModel", DummyModel)

    spec = ModelSpec(
        base_url="http://base", model_name="gpt-test", api_key_env="TEST_KEY"
    )
    model = build_model(spec)

    assert isinstance(model, DummyModel)
    assert model.model_name == "gpt-test"
    assert model.provider.base_url == "http://base"
    assert model.provider.api_key == "abc"


def test_resolve_schema_valid_missing_and_invalid() -> None:
    schema_module = types.ModuleType("schemas.orch")

    class GoodSchema(BaseModel):
        x: int

    class NotSchema:
        pass

    schema_module.__dict__["GoodSchema"] = GoodSchema
    schema_module.__dict__["NotSchema"] = NotSchema
    sys.modules["schemas.orch"] = schema_module

    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[ChainStepSpec(id="s1", kind="text", prompt_section="step:one")],
        schemas={"Good": "schemas.orch:GoodSchema", "Bad": "schemas.orch:NotSchema"},
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec, instructions="", system_prompt="", step_prompts={}
    )

    try:
        assert resolve_schema(loaded, "Good") is GoodSchema
        with pytest.raises(KeyError):
            resolve_schema(loaded, "Missing")
        with pytest.raises(TypeError):
            resolve_schema(loaded, "Bad")
    finally:
        sys.modules.pop("schemas.orch", None)


def test_build_toolset_filters_agent_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sentinel_toolset = RecordingToolset()
    recorder = LoadToolsRecorder(sentinel_toolset)
    monkeypatch.setattr(tools_module, "load_tools", recorder)
    tools = AgentTools([tmp_path])
    toolset = tools.build_toolset(
        ["tool.a", "agent_call", "tool.b"], _permissions(tmp_path)
    )

    assert toolset is sentinel_toolset
    assert len(recorder.calls) == 1
    roots, tool_ids, perms = recorder.calls[0]
    assert roots == [tmp_path]
    assert tool_ids == ["tool.a", "tool.b"]
    assert perms.root == _permissions(tmp_path).root


def test_build_toolset_returns_empty_for_agent_call_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recorder = LoadToolsRecorder(RecordingToolset())
    monkeypatch.setattr(tools_module, "load_tools", recorder)
    tools = AgentTools([tmp_path])
    toolset = tools.build_toolset(["agent_call"], _permissions(tmp_path))

    assert isinstance(toolset, FunctionToolset)
    assert recorder.calls == []


def test_maybe_inject_agent_call_tool_adds_when_requested() -> None:
    tools = AgentTools([])
    toolset = RecordingToolset()

    tools.maybe_inject_agent_call(
        ["agent_call"],
        toolset,
        "run/input.json",
        AsyncCallRecorder(return_value={"text": "ok"}),
    )

    assert len(toolset.add_calls) == 1
    func, name, description = toolset.add_calls[0]
    assert hasattr(func, "__self__")
    assert isinstance(func.__self__, AgentCallTool)
    assert name == "agent_call"
    assert "another agent" in description


def test_maybe_inject_agent_call_tool_skips_when_missing() -> None:
    tools = AgentTools([])
    toolset = RecordingToolset()

    tools.maybe_inject_agent_call(
        [],
        toolset,
        "run/input.json",
        AsyncCallRecorder(return_value={"text": "ok"}),
    )

    assert toolset.add_calls == []


@pytest.mark.anyio
async def test_agent_call_tool_accepts_input_text() -> None:
    recorder = AsyncCallRecorder(return_value="ok")
    tool = AgentCallTool(recorder, "run/input.json")

    result = await tool(agent="child", input_text="hello")

    assert result == {"text": "ok"}
    assert len(recorder.calls) == 1
    args = recorder.calls[0]["args"]
    assert args[0] == "child"
    assert isinstance(args[1], input_adaptors_module.TextInput)
    run_input = args[1].load()
    assert run_input.kind == "text"
    assert run_input.text == "hello"


@pytest.mark.anyio
async def test_agent_call_tool_rejects_missing_or_duplicate_input() -> None:
    recorder = AsyncCallRecorder(return_value="ok")
    tool = AgentCallTool(recorder, "run/input.json")

    with pytest.raises(ValueError):
        await tool(agent="child")
    with pytest.raises(ValueError):
        await tool(agent="child", input_file="a.txt", input_text="hi")


def test_init_state_contains_agent_id_and_steps() -> None:
    qa = _make_quick_agent_for_test()

    qa._agent_id = "agent-1"
    state = qa._init_state()

    assert state == {"agent_id": "agent-1", "steps": {}, "last_step_output": None}


def test_build_model_settings_openai_compatible() -> None:
    qa = _make_quick_agent_for_test()
    spec = ModelSpec(base_url="http://x", model_name="m", provider="openai-compatible")

    settings = qa._build_model_settings(spec)

    assert settings == {"extra_body": {"format": "json"}}


def test_build_model_settings_openai_endpoint_skips_format() -> None:
    qa = _make_quick_agent_for_test()
    spec = ModelSpec(
        base_url="https://api.openai.com/v1",
        model_name="m",
        provider="openai-compatible",
    )

    settings = qa._build_model_settings(spec)

    assert settings is None


def test_build_model_settings_other_provider() -> None:
    qa = _make_quick_agent_for_test()
    spec = ModelSpec(base_url="http://x", model_name="m", provider="other")

    settings = qa._build_model_settings(spec)

    assert settings is None


def test_build_model_settings_includes_extra_body() -> None:
    qa = _make_quick_agent_for_test()
    spec = ModelSpec(
        base_url="https://api.openai.com/v1",
        model_name="m",
        provider="openai-compatible",
    )

    settings = qa._build_model_settings(spec)

    assert settings is None

    qa = _make_quick_agent_for_test()
    spec = ModelSpec(
        base_url="https://api.openai.com/v1",
        model_name="m",
        provider="openai-compatible",
    )
    qa.extra_body = {"foo": "bar"}

    settings = qa._build_model_settings(spec)

    assert settings == {"extra_body": {"foo": "bar"}}


def test_build_model_settings_openai_endpoint_strips_num_ctx() -> None:
    qa = _make_quick_agent_for_test()
    spec = ModelSpec(
        base_url="https://api.openai.com/v1",
        model_name="m",
        provider="openai-compatible",
    )
    qa.extra_body = {"options": {"num_ctx": 8192, "other": 1}}

    settings = qa._build_model_settings(spec)

    assert settings == {"extra_body": {"options": {"other": 1}}}


def test_build_model_settings_openai_endpoint_strips_num_ctx_all_removed() -> None:
    qa = _make_quick_agent_for_test()
    spec = ModelSpec(
        base_url="https://api.openai.com/v1",
        model_name="m",
        provider="openai-compatible",
    )
    qa.extra_body = {"options": {"num_ctx": 8192}}

    settings = qa._build_model_settings(spec)

    assert settings is None


def test_build_structured_model_settings_non_openai_passthrough() -> None:
    qa = _make_quick_agent_for_test()
    schema = ExampleSchema
    settings: ModelSettings = {"extra_body": {"format": "json"}}
    qa.model = cast(OpenAIChatModel, DummyOpenAIModel("http://localhost"))
    qa.model_settings_json = settings

    result = qa._build_structured_model_settings(schema_cls=schema)

    assert result == settings


def test_build_structured_model_settings_openai_injects_schema() -> None:
    qa = _make_quick_agent_for_test()
    schema = ExampleSchema
    qa.model = cast(OpenAIChatModel, DummyOpenAIModel("https://api.openai.com/v1"))

    result = qa._build_structured_model_settings(schema_cls=schema)

    assert result is not None
    extra_body_obj = result.get("extra_body")
    assert extra_body_obj is not None
    assert isinstance(extra_body_obj, dict)
    response_format_obj = extra_body_obj["response_format"]
    assert isinstance(response_format_obj, dict)
    assert response_format_obj["type"] == "json_schema"
    json_schema_obj = response_format_obj["json_schema"]
    assert isinstance(json_schema_obj, dict)
    assert json_schema_obj["name"] == "ExampleSchema"
    assert json_schema_obj["strict"] is True


def test_build_user_prompt_uses_prompting(monkeypatch: pytest.MonkeyPatch) -> None:
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)
    recorder = SyncCallRecorder(return_value="prompt")
    monkeypatch.setattr(qa_module, "make_user_prompt", recorder)

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.run_input = run_input
    qa.state = {"agent_id": "agent-1", "steps": {}, "last_step_output": None}

    result = qa_module.make_user_prompt(run_input, qa.state)

    assert result == "prompt"
    assert recorder.calls == [
        (
            (run_input, {"agent_id": "agent-1", "steps": {}, "last_step_output": None}),
            {},
        )
    ]


@pytest.mark.anyio
async def test_run_text_step_raises_for_missing_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:missing")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    with pytest.raises(KeyError):
        await qa._run_text_step(
            step=step,
        )


@pytest.mark.anyio
async def test_run_step_text_returns_output(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> str:
        captured["batch_request"] = batch_request
        return "hello"

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
    output = await qa._run_step(
        step=step,
    )

    assert output == "hello"
    batch_request = captured["batch_request"]
    assert batch_request.messages[0].content == "systemdo thing"
    assert batch_request.messages[1].content == make_user_prompt(run_input, qa.state)
    assert batch_request.model.model_name == "m"


@pytest.mark.anyio
async def test_run_text_step_omits_tools_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> str:
        captured["batch_request"] = batch_request
        return "hello"

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
    qa.tool_ids = []

    output = await qa._run_text_step(
        step=step,
    )

    assert output == "hello"
    assert qa._toolsets_for_run() == []


@pytest.mark.anyio
async def test_run_step_structured_parses_json_with_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> ExampleSchema:
        captured["batch_request"] = batch_request
        return ExampleSchema(x=7)

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    schema_module = types.ModuleType("schemas.struct")
    schema_module.__dict__["ExampleSchema"] = ExampleSchema
    sys.modules["schemas.struct"] = schema_module

    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Example"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        schemas={"Example": "schemas.struct:ExampleSchema"},
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    run_input = RunInput(source_path="in.json", kind="json", text="{}", data={})

    try:
        qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
        qa.loaded = loaded
        qa.model = cast(OpenAIChatModel, object())
        qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
        qa.model_settings_json = {"extra_body": {"format": "json"}}
        qa.toolset = RecordingToolset()
        qa.tool_ids = []
        qa.run_input = run_input
        qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
        output = await qa._run_step(
            step=step,
        )
    finally:
        sys.modules.pop("schemas.struct", None)

    assert isinstance(output, ExampleSchema)
    assert output.x == 7
    assert captured["batch_request"].step_kind == "structured"
    assert captured["batch_request"].response_format is not None


@pytest.mark.anyio
async def test_run_step_unknown_kind_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    step = ChainStepSpec(id="s1", kind="mystery", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
    with pytest.raises(NotImplementedError):
        await qa._run_step(
            step=step,
        )


@pytest.mark.anyio
async def test_run_text_step_uses_make_user_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> str:
        captured["batch_request"] = batch_request
        return "ok"

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    recorder = SyncCallRecorder(return_value="prompt")
    monkeypatch.setattr(qa_module, "make_user_prompt", recorder)

    output = await qa._run_text_step(
        step=step,
    )

    assert output == "ok"
    assert captured["batch_request"].messages[-1].content == "prompt"
    assert recorder.calls == [
        (
            (run_input, qa.state),
            {},
        )
    ]


@pytest.mark.anyio
async def test_run_text_step_no_instructions_or_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> str:
        captured["batch_request"] = batch_request
        return "ok"

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    output = await qa._run_text_step(
        step=step,
    )

    assert output == "ok"
    assert captured["batch_request"].messages[0].content == "do thing"
    assert captured["batch_request"].messages[1].content == make_user_prompt(run_input, qa.state)


@pytest.mark.anyio
async def test_run_text_step_system_prompt_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> str:
        captured["batch_request"] = batch_request
        return "ok"

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="",
        system_prompt="You are concise.",
        step_prompts={"step:one": "do thing"},
    )
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    output = await qa._run_text_step(
        step=step,
    )

    assert output == "ok"
    assert captured["batch_request"].messages[0].content == "You are concise.\ndo thing"
    assert captured["batch_request"].messages[1].content == make_user_prompt(run_input, qa.state)


@pytest.mark.anyio
async def test_run_text_step_instructions_only(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> str:
        captured["batch_request"] = batch_request
        return "ok"

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="Use the tool.",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    output = await qa._run_text_step(
        step=step,
    )

    assert output == "ok"
    assert captured["batch_request"].messages[0].content == "Use the tool.do thing"
    assert captured["batch_request"].messages[1].content == make_user_prompt(run_input, qa.state)


@pytest.mark.anyio
async def test_run_text_step_logs_llm_request_payload_immediately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> str:
        captured["batch_request"] = batch_request
        return "ok"

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )
    monkeypatch.chdir(tmp_path)

    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step], output=OutputSpec(file=None))
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    log_path = tmp_path / "log" / "custom.log"
    qa = _make_quick_agent_for_test(
        loaded=loaded,
        run_input=run_input,
        enable_llm_request_logging=True,
        llm_log_path=log_path,
    )
    qa._agent_id = "agent-1"
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa._recorder._agent_id = qa._agent_id
    qa._recorder.model_spec = qa.model_spec
    qa._recorder.effective_base_url = qa._effective_base_url()
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa._recorder.tool_ids = list(qa.tool_ids)
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    await qa._run_text_step(
        step=step,
    )

    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    entries = [entry for entry in log_text.split("[LLM_REQUEST]\n") if entry.strip()]
    assert len(entries) == 1
    entry_payload = json.loads(entries[0])

    assert entry_payload["request_state"] == "before_request_start"
    assert entry_payload["agent_id"] == "agent-1"
    assert entry_payload["step"]["id"] == "s1"
    assert entry_payload["step"]["kind"] == "text"
    assert entry_payload["instructions"] == "systemdo thing"
    assert entry_payload["user_prompt"] == make_user_prompt(run_input, qa.state)
    assert '"timestamp_utc": "' in entries[0]


@pytest.mark.anyio
async def test_run_structured_step_missing_schema_raises() -> None:
    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema=None
    )
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.json", kind="json", text="{}", data={})

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.run_input = run_input
    with pytest.raises(ValueError):
        await qa._run_structured_step(
            step=step,
        )


@pytest.mark.anyio
async def test_run_structured_step_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> ExampleSchema:
        captured["batch_request"] = batch_request
        return ExampleSchema(x=3)

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    schema_module = types.ModuleType("schemas.struct2")
    schema_module.__dict__["ExampleSchema"] = ExampleSchema
    sys.modules["schemas.struct2"] = schema_module

    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Example"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        schemas={"Example": "schemas.struct2:ExampleSchema"},
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    run_input = RunInput(source_path="in.json", kind="json", text="{}", data={})

    try:
        qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
        qa.loaded = loaded
        qa.model = cast(OpenAIChatModel, object())
        qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
        qa.toolset = RecordingToolset()
        qa.tool_ids = []
        qa.run_input = run_input
        qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
        output = await qa._run_structured_step(
            step=step,
        )
    finally:
        sys.modules.pop("schemas.struct2", None)

    assert output.model_dump() == {"x": 3}
    assert isinstance(output, ExampleSchema)
    assert captured["batch_request"].step_kind == "structured"


@pytest.mark.anyio
async def test_run_structured_step_adds_json_schema_for_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> ExampleSchema:
        captured["batch_request"] = batch_request
        return ExampleSchema(x=9)

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    schema_module = types.ModuleType("schemas.struct3")
    schema_module.__dict__["ExampleSchema"] = ExampleSchema
    sys.modules["schemas.struct3"] = schema_module

    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Example"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="https://api.openai.com/v1", model_name="m"),
        chain=[step],
        schemas={"Example": "schemas.struct3:ExampleSchema"},
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    run_input = RunInput(source_path="in.json", kind="json", text="{}", data={})

    try:
        qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
        qa.loaded = loaded
        qa.model = cast(OpenAIChatModel, DummyOpenAIModel("https://api.openai.com/v1"))
        qa.model_spec = ModelSpec(base_url="https://api.openai.com/v1", model_name="m")
        qa.toolset = RecordingToolset()
        qa.tool_ids = []
        qa.run_input = run_input
        qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
        await qa._run_structured_step(
            step=step,
        )
    finally:
        sys.modules.pop("schemas.struct3", None)

    assert captured["batch_request"].response_format is not None
    response_format = captured["batch_request"].response_format
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "ExampleSchema"
    assert response_format["json_schema"]["strict"] is True


@pytest.mark.anyio
async def test_run_chain_updates_state_and_returns_last() -> None:
    step1 = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    step2 = ChainStepSpec(id="s2", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step1, step2])

    qa = RecordingQuickAgent(outputs=[({"a": 1}, "first"), ("b", "second")])
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    final = await qa._run_chain()

    assert final == "b"
    assert qa.state["steps"] == {"s1": {"a": 1}, "s2": "b"}
    assert qa.state["last_step_output"] == "b"
    assert qa.calls == ["s1", "s2"]


@pytest.mark.anyio
async def test_run_returns_compiled_json_output_when_enabled() -> None:
    step1 = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    step2 = ChainStepSpec(id="s2", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step1, step2])
    loaded.spec.output.return_compiled_output = True

    qa = _make_quick_agent_for_test(loaded=loaded)
    qa.state = {
        "agent_id": "a",
        "steps": {"s1": {"a": 1}, "s2": "b"},
        "last_step_output": "b",
    }

    async def fake_run_chain() -> str:
        return "b"

    qa._run_chain = fake_run_chain  # type: ignore[assignment]

    output = await qa.run()

    assert output == {"s1": {"a": 1}, "s2": "b", "last_step_output": "b"}


@pytest.mark.anyio
async def test_run_returns_compiled_text_output_when_enabled() -> None:
    step1 = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    step2 = ChainStepSpec(id="s2", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step1, step2])
    loaded.spec.output.return_compiled_output = True
    loaded.spec.output.format = "markdown"

    qa = _make_quick_agent_for_test(loaded=loaded)
    qa.state = {
        "agent_id": "a",
        "steps": {"s1": "first", "s2": "second"},
        "last_step_output": "second",
    }

    async def fake_run_chain() -> str:
        return "second"

    qa._run_chain = fake_run_chain  # type: ignore[assignment]

    output = await qa.run()

    assert output == "first\nsecond"


@pytest.mark.anyio
async def test_run_returns_compiled_structured_output_when_enabled() -> None:
    # Chain defines 3 steps but compiled output schema only includes step2 and step3.
    step1 = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    step2 = ChainStepSpec(id="s2", kind="text", prompt_section="step:one")
    step3 = ChainStepSpec(id="s3", kind="text", prompt_section="step:one")

    class FinalOutput(BaseModel):
        s2: str
        s3: str

    schema_module = types.ModuleType("schemas.compiled")
    schema_module.__dict__["FinalOutput"] = FinalOutput
    sys.modules["schemas.compiled"] = schema_module

    output_spec = OutputSpec.model_validate(
        {
            "file": "out/result.json",
            "format": "structured",
            "return_compiled_output": True,
            "schema": "Final",
        }
    )

    loaded = _make_loaded_with_chain(
        [step1, step2, step3],
        schemas={"Final": "schemas.compiled:FinalOutput"},
        output=output_spec,
    )

    qa = _make_quick_agent_for_test(loaded=loaded)
    qa.state = {
        "agent_id": "a",
        "steps": {"s2": "two", "s3": "three"},
        "last_step_output": "three",
    }

    async def fake_run_chain() -> str:
        return "three"

    qa._run_chain = fake_run_chain  # type: ignore[assignment]

    output = await qa.run()

    assert isinstance(output, FinalOutput)
    assert output.model_dump() == {"s2": "two", "s3": "three"}

    sys.modules.pop("schemas.compiled", None)


@pytest.mark.anyio
async def test_run_returns_compiled_output_with_missing_step_keys() -> None:
    # Even if the chain defines 3 steps, compiled output only reflects steps present in state.
    step1 = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    step2 = ChainStepSpec(id="s2", kind="text", prompt_section="step:one")
    step3 = ChainStepSpec(id="s3", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step1, step2, step3])
    loaded.spec.output.return_compiled_output = True

    qa = _make_quick_agent_for_test(loaded=loaded)
    qa.state = {
        "agent_id": "a",
        "steps": {"s1": "one", "s2": "two"},
        "last_step_output": "two",
    }

    async def fake_run_chain() -> str:
        return "two"

    qa._run_chain = fake_run_chain  # type: ignore[assignment]

    output = await qa.run()

    assert output == {"s1": "one", "s2": "two", "last_step_output": "two"}


def test_apply_sample_processing_updates_input_text() -> None:
    loaded = _make_loaded_with_chain(
        [ChainStepSpec(id="s1", kind="text", prompt_section="step:one")]
    )
    loaded.spec.content_processing = ContentProcessingSpec(
        sample=SampleSpec(ratios=(100, 0, 0), max_chunk_tokens=3)
    )
    run_input = RunInput(
        source_path="in.txt",
        kind="text",
        text="one two three four five six",
        data=None,
    )
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)

    qa._apply_sample_processing()

    assert qa.run_input.text == "one two three"


def test_apply_sample_processing_keeps_text_when_not_configured() -> None:
    loaded = _make_loaded_with_chain(
        [ChainStepSpec(id="s1", kind="text", prompt_section="step:one")]
    )
    run_input = RunInput(
        source_path="in.txt",
        kind="text",
        text="one two three four five six",
        data=None,
    )
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)

    qa._apply_sample_processing()

    assert qa.run_input.text == "one two three four five six"


def test_apply_sample_processing_writes_debug_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _make_loaded_with_chain(
        [ChainStepSpec(id="s1", kind="text", prompt_section="step:one")]
    )
    loaded.spec.content_processing = ContentProcessingSpec(
        sample=SampleSpec(
            ratios=(100, 0, 0),
            max_chunk_tokens=3,
            debug_output_file="out/sample_debug.txt",
        )
    )
    run_input = RunInput(
        source_path="in.txt",
        kind="text",
        text="one two three four five six",
        data=None,
    )
    write_output_recorder = SyncCallRecorder(return_value=None)
    monkeypatch.setattr(qa_module, "write_text_output", write_output_recorder)
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)

    qa._apply_sample_processing()

    assert len(write_output_recorder.calls) == 1
    args = write_output_recorder.calls[0][0]
    assert args[0] == Path("out/sample_debug.txt")
    assert args[1] == "one two three"


@pytest.mark.anyio
async def test_run_returns_sample_output_without_llm_for_empty_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[],
        output=OutputSpec(file=None),
        content_processing=ContentProcessingSpec(
            sample=SampleSpec(ratios=(100, 0, 0), max_chunk_tokens=3)
        ),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="",
        system_prompt="",
        step_prompts={},
    )
    run_input = RunInput(
        source_path="in.txt",
        kind="text",
        text="one two three four five six",
        data=None,
    )
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    run_chain_recorder = AsyncCallRecorder(return_value="unexpected")
    monkeypatch.setattr(qa, "_run_chain", run_chain_recorder)

    output = await qa.run()

    assert output == "one two three"
    assert len(run_chain_recorder.calls) == 0


@pytest.mark.anyio
async def test_apply_chunk_processing_runs_chain_per_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _make_loaded_with_chain(
        [ChainStepSpec(id="s1", kind="text", prompt_section="step:one")]
    )
    loaded.spec.content_processing = ContentProcessingSpec(
        chunk_processing=ChunkProcessingSpec(
            mode="map_chunks",
            provider="semchunks",
            max_chunk_tokens=3,
            overlap_percent=0,
        )
    )
    run_input = RunInput(
        source_path="in.txt",
        kind="text",
        text="one two three four five six seven eight",
        data=None,
    )
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)

    run_chunk_agent_recorder = AsyncCallRecorder(return_value="processed")
    monkeypatch.setattr(qa, "_run_chunk_agent", run_chunk_agent_recorder)

    result = await qa._apply_chunk_processing()

    assert result is not None
    assert isinstance(result, list)
    items = result
    assert len(items) >= 2
    assert all(entry == "processed" for entry in items)
    assert len(run_chunk_agent_recorder.calls) == len(items)


@pytest.mark.anyio
async def test_run_returns_chunk_text_items_without_llm_for_empty_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[],
        output=OutputSpec(file=None),
        content_processing=ContentProcessingSpec(
            chunk_processing=ChunkProcessingSpec(
                mode="map_chunks",
                provider="semchunks",
                max_chunk_tokens=3,
                overlap_percent=0,
            )
        ),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="",
        system_prompt="",
        step_prompts={},
    )
    run_input = RunInput(
        source_path="in.txt",
        kind="text",
        text="one two three four five six seven eight",
        data=None,
    )
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    run_chunk_agent_recorder = AsyncCallRecorder(return_value="processed")
    monkeypatch.setattr(qa, "_run_chunk_agent", run_chunk_agent_recorder)

    output = await qa.run()

    assert isinstance(output, list)
    items = output
    assert len(items) >= 2
    assert all(isinstance(entry, str) for entry in items)
    assert len(run_chunk_agent_recorder.calls) == 0


@pytest.mark.anyio
async def test_run_returns_chunk_output_when_chunk_processing_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _make_loaded_with_chain(
        [ChainStepSpec(id="s1", kind="text", prompt_section="step:one")]
    )
    loaded.spec.content_processing = ContentProcessingSpec(
        chunk_processing=ChunkProcessingSpec(
            mode="map_paragraphs",
            provider="semchunks",
            max_chunk_tokens=8,
            overlap_percent=0,
        )
    )
    run_input = RunInput(
        source_path="in.txt",
        kind="text",
        text="p1 one two three.\n\np2 four five six.\n\np3 seven eight nine.",
        data=None,
    )
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    run_chunk_agent_recorder = AsyncCallRecorder(return_value="processed")
    monkeypatch.setattr(qa, "_run_chunk_agent", run_chunk_agent_recorder)

    output = await qa.run()

    assert isinstance(output, list)
    items = output
    assert len(items) >= 1
    assert all(entry == "processed" for entry in items)
    assert len(run_chunk_agent_recorder.calls) == len(items)


def test_run_chunk_processing_raises_for_invalid_provider() -> None:
    qa = _make_quick_agent_for_test()
    map_config = ChunkProcessingSpec.model_construct(
        mode="map_chunks",
        provider="invalid",
        max_chunk_tokens=4,
        overlap_percent=0,
        overlap_token=None,
    )

    with pytest.raises(ValueError, match="chunk_processing.provider"):
        qa._run_chunk_processing("one two three four", map_config)


@pytest.mark.anyio
async def test_run_chain_single_shot_system_prompt_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> str:
        captured["batch_request"] = batch_request
        return "hello"

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[],
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="",
        system_prompt="You are concise.",
        step_prompts={},
    )
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, DummyOpenAIModel("http://x"))
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    output = await qa._run_chain()

    assert output == "hello"
    assert captured["batch_request"].messages[0].content == "You are concise."
    assert captured["batch_request"].messages[1].content == make_user_prompt(run_input, qa.state)


@pytest.mark.anyio
async def test_run_chain_single_shot_instructions_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> str:
        captured["batch_request"] = batch_request
        return "hello"

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[],
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="Use the tool.",
        system_prompt="",
        step_prompts={},
    )
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, DummyOpenAIModel("http://x"))
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    output = await qa._run_chain()

    assert output == "hello"
    assert captured["batch_request"].messages[0].content == "Use the tool."
    assert captured["batch_request"].messages[1].content == make_user_prompt(run_input, qa.state)


@pytest.mark.anyio
async def test_run_text_step_maps_tools_not_supported_to_quick_agent_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            raise qa_module.openai.APIStatusError(
                "tools unsupported",
                response=httpx.Response(
                    status_code=400,
                    request=httpx.Request(
                        method="POST",
                        url="http://x/chat/completions",
                        headers={"Content-Type": "application/json"},
                        content=b'{}',
                    ),
                    headers={"x-response-id": "resp-1"},
                    content=b'{"message":"registry.ollama.ai/library/deepseek-r1:14b does not support tools"}',
                ),
                body={
                    "message": "registry.ollama.ai/library/deepseek-r1:14b does not support tools"
                },
            )

    class FakeAsyncOpenAI:
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str,
            timeout: float | None,
            http_client: Any,
        ) -> None:
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(qa_module.openai, "AsyncOpenAI", FakeAsyncOpenAI)

    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.json", kind="json", text="{}", data={})
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, types.SimpleNamespace(client=None))
    qa.model_spec = ModelSpec(base_url="http://x", model_name="deepseek-r1:14b")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
    with pytest.raises(QuickAgentToolsNotSupportedException):
        await qa._run_text_step(step=step)


@pytest.mark.anyio
async def test_run_single_shot_maps_chat_not_supported_to_quick_agent_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompletions:
        async def create(self, **kwargs: Any) -> Any:
            raise qa_module.openai.APIStatusError(
                "chat unsupported",
                response=httpx.Response(
                    status_code=400,
                    request=httpx.Request(
                        method="POST",
                        url="http://x/chat/completions",
                        headers={"Content-Type": "application/json"},
                        content=b'{}',
                    ),
                    headers={"x-response-id": "resp-1"},
                    content=b'{"message":"\"nomic-embed-text:v1.5\" does not support chat"}',
                ),
                body={"message": '"nomic-embed-text:v1.5" does not support chat'},
            )

    class FakeAsyncOpenAI:
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str,
            timeout: float | None,
            http_client: Any,
        ) -> None:
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(qa_module.openai, "AsyncOpenAI", FakeAsyncOpenAI)

    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="nomic-embed-text:v1.5"),
        chain=[],
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="Use the tool.",
        system_prompt="",
        step_prompts={},
    )
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, types.SimpleNamespace(client=None))
    qa.model_spec = ModelSpec(base_url="http://x", model_name="nomic-embed-text:v1.5")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
    with pytest.raises(QuickAgentChatNotSupportedException):
        await qa._run_single_shot()


@pytest.mark.anyio
async def test_run_single_shot_structured_uses_schema_output_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    usage_payload = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}

    class ForbiddenAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise AssertionError(
                "pydantic_ai.Agent should not be used for structured single-shot mode."
            )

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> OpenAIResponseStub:
            captured["create_kwargs"] = kwargs
            return OpenAIResponseStub('{"msg":"ok"}', usage=usage_payload)

    class FakeAsyncOpenAI:
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str,
            timeout: float | None,
            http_client: Any,
        ) -> None:
            captured["init_kwargs"] = {
                "api_key": api_key,
                "base_url": base_url,
                "timeout": timeout,
                "http_client": http_client,
            }
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(single_shot_module, "Agent", ForbiddenAgent)
    monkeypatch.setattr(qa_module.openai, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(
            base_url="https://api.openai.com/v1", model_name="gpt-4.1-mini"
        ),
        chain=[],
        schemas={"Output": "test_orchestrator:OutputSchema"},
        output=OutputSpec(file=None, output_schema="Output"),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="Return structured output.",
        system_prompt="",
        step_prompts={},
    )
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, types.SimpleNamespace(client=None))
    qa.model_spec = spec.model
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    output = await qa._run_single_shot()

    assert isinstance(output, OutputSchema)
    assert output.msg == "ok"
    init_kwargs = captured["init_kwargs"]
    assert init_kwargs["api_key"] == "test-key"
    assert init_kwargs["base_url"] == "https://api.openai.com/v1"
    create_kwargs = captured["create_kwargs"]
    assert create_kwargs["model"] == "gpt-4.1-mini"
    response_format = create_kwargs["extra_body"]["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "OutputSchema"
    assert response_format["json_schema"]["strict"] is True
    assert qa.last_run_metrics == {
        "provider": "openai-compatible",
        "model": "gpt-4.1-mini",
        "usage": usage_payload,
        "completion_id": "cmpl-123",
        "created": 123,
        "system_fingerprint": "fp-1",
        "finish_reason": "stop",
    }


@pytest.mark.anyio
async def test_run_single_shot_structured_passes_timeout_to_openai_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeCompletions:
        async def create(self, **kwargs: Any) -> OpenAIResponseStub:
            _ = kwargs
            return OpenAIResponseStub('{"msg":"ok"}')

    class FakeAsyncOpenAI:
        def __init__(
            self, *, api_key: str, base_url: str, timeout: float, http_client: Any
        ) -> None:
            captured["init_kwargs"] = {
                "api_key": api_key,
                "base_url": base_url,
                "timeout": timeout,
                "http_client": http_client,
            }
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(single_shot_module.openai, "AsyncOpenAI", FakeAsyncOpenAI)
    spec = AgentSpec(
        name="test",
        model=ModelSpec(
            base_url="https://api.openai.com/v1",
            model_name="gpt-5.2",
            timeout_seconds=77.0,
        ),
        chain=[],
        schemas={"Output": "test_orchestrator:OutputSchema"},
        output=OutputSpec(file=None, output_schema="Output"),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="Return structured output.",
        system_prompt="",
        step_prompts={},
    )
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, types.SimpleNamespace(client=None))
    qa.model_spec = spec.model
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    output = await qa._run_single_shot()

    assert isinstance(output, OutputSchema)
    init_kwargs = captured["init_kwargs"]
    assert init_kwargs["timeout"] == 77.0


@pytest.mark.anyio
async def test_run_single_shot_structured_parses_json_with_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompletions:
        async def create(self, **kwargs: Any) -> OpenAIResponseStub:
            _ = kwargs
            return OpenAIResponseStub('preface {"msg":"ok"} suffix')

    class FakeAsyncOpenAI:
        def __init__(
            self,
            *,
            api_key: str,
            base_url: str,
            timeout: float | None,
            http_client: Any,
        ) -> None:
            _ = (api_key, base_url, timeout, http_client)
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(single_shot_module.openai, "AsyncOpenAI", FakeAsyncOpenAI)
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[],
        schemas={"Output": "test_orchestrator:OutputSchema"},
        output=OutputSpec(file=None, output_schema="Output"),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="Return structured output.",
        system_prompt="",
        step_prompts={},
    )
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, types.SimpleNamespace(client=None))
    qa.model_spec = spec.model
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    output = await qa._run_single_shot()

    assert isinstance(output, OutputSchema)
    assert output.msg == "ok"


@pytest.mark.anyio
async def test_run_single_shot_structured_rejects_tools() -> None:
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[],
        tools=["agent_call"],
        schemas={"Output": "test_orchestrator:OutputSchema"},
        output=OutputSpec(file=None, output_schema="Output"),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="Return structured output.",
        system_prompt="",
        step_prompts={},
    )
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    async def batch_call(request: BatchSubmitRequest) -> BatchImportRequest:
        assert request.output_schema == "Output"
        raise ValueError("output.output_schema does not support tools")

    qa = _make_quick_agent_for_test(
        loaded=loaded,
        run_input=run_input,
        memory={"batch_call": batch_call},
    )
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, types.SimpleNamespace(client=None))
    qa.model_spec = spec.model
    qa.run_input = run_input
    qa.tool_ids = ["agent_call"]
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
    with pytest.raises(ValueError, match="output.output_schema does not support tools"):
        await qa._run_single_shot()


@pytest.mark.anyio
async def test_run_single_shot_structured_still_uses_batch_call_when_flag_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[],
        single_shot_use_pydantic_ai=True,
        schemas={"Output": "test_orchestrator:OutputSchema"},
        output=OutputSpec(file=None, output_schema="Output"),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="Return structured output.",
        system_prompt="",
        step_prompts={},
    )
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    async def batch_call(request: BatchSubmitRequest) -> BatchImportRequest:
        return BatchImportRequest(
            request_id=request.request_id,
            payload={"state": "completed", "output": {"msg": "ok"}},
        )

    qa = _make_quick_agent_for_test(
        loaded=loaded,
        run_input=run_input,
        memory={"batch_call": batch_call},
    )
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, types.SimpleNamespace(client=None))
    qa.model_spec = spec.model
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}

    output = await qa._run_single_shot()

    assert isinstance(output, OutputSchema)
    assert output.msg == "ok"


@pytest.mark.anyio
async def test_run_text_step_propagates_unexpected_model_behavior_with_request_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request(
        method="POST",
        url="http://localhost:11434/v1/chat/completions",
        headers={"x-test-header": "abc"},
        content=b'{"messages":[{"role":"user","content":"hello"}]}',
    )
    response = httpx.Response(
        status_code=500,
        request=request,
        headers={"x-response-id": "resp-1"},
        content=b'{"error":"internal"}',
    )
    cause = httpx.HTTPStatusError("Server error", request=request, response=response)
    unexpected_error = UnexpectedModelBehavior(
        "Unexpected response", body='{"error":"internal"}'
    )
    unexpected_error.__cause__ = cause

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> str:
        raise unexpected_error

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.json", kind="json", text="{}", data={})
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
    with pytest.raises(UnexpectedModelBehavior) as raised:
        await qa._run_text_step(step=step)
    assert raised.value is unexpected_error
    cause_obj = raised.value.__cause__
    assert isinstance(cause_obj, httpx.HTTPStatusError)
    request_details = {
        "method": cause_obj.request.method,
        "url": str(cause_obj.request.url),
        "headers": dict(cause_obj.request.headers),
        "body": cause_obj.request.content.decode("utf-8"),
    }
    assert request_details == {
        "method": "POST",
        "url": "http://localhost:11434/v1/chat/completions",
        "headers": {
            "host": "localhost:11434",
            "x-test-header": "abc",
            "content-length": "48",
        },
        "body": '{"messages":[{"role":"user","content":"hello"}]}',
    }
    response_details = {
        "status_code": cause_obj.response.status_code,
        "headers": dict(cause_obj.response.headers),
        "body": cause_obj.response.content.decode("utf-8"),
    }
    assert response_details == {
        "status_code": 500,
        "headers": {
            "x-response-id": "resp-1",
            "content-length": "20",
        },
        "body": '{"error":"internal"}',
    }
    assert qa._recorder.execution_log
    assert qa._recorder.execution_log[-1].call_site == "run_text_step"
    curl_command = qa._recorder.execution_log[-1].to_curl()
    assert "curl -X POST" in curl_command
    assert "-H 'Content-Type: application/json'" in curl_command
    assert '"model": "m"' in curl_command
    assert "http://x/chat/completions" in curl_command


def test_execution_log_entry_to_curl_reconstructs_when_request_missing() -> None:
    entry = ExecutionLogEntry(
        call_site="test",
        request_context={
            "base_url": "http://localhost:11434/v1",
            "model_name": "llama3",
            "instructions": "Use tools if needed.",
            "system_prompt": "You are concise.",
            "user_prompt": "Summarize this file.",
            "model_settings": {"extra_body": {"format": "json"}},
        },
    )
    curl_command = entry.to_curl()
    assert "curl -X POST" in curl_command
    assert "http://localhost:11434/v1/chat/completions" in curl_command
    assert "-H 'Content-Type: application/json'" in curl_command
    assert '"model": "llama3"' in curl_command
    assert '"format": "json"' in curl_command


def test_record_llm_request_uses_client_base_url_for_execution_log() -> None:
    qa = _make_quick_agent_for_test()
    qa.model_spec = ModelSpec(base_url="http://localhost:11434", model_name="m")
    qa._recorder.model_spec = qa.model_spec
    qa._recorder.effective_base_url = qa._effective_base_url()
    qa._recorder._record_llm_request(
        call_site="run_single_shot",
        step_id=None,
        step_kind="single_shot",
        output_schema=None,
        instructions="Use tools if needed.",
        system_prompt="You are concise.",
        user_prompt="Summarize this file.",
        model_settings=ModelSettings(extra_body={"format": "json"}),
    )
    assert qa._recorder.execution_log
    curl_command = qa._recorder.execution_log[-1].to_curl()
    assert "http://localhost:11434/chat/completions" in curl_command
    assert "http://localhost:11434/v1/chat/completions" not in curl_command


@pytest.mark.anyio
async def test_run_text_step_unexpected_model_behavior_uses_last_http_log_entry_for_curl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unexpected_error = UnexpectedModelBehavior(
        "Exceeded maximum retries (1) for output validation"
    )

    async def fake_execute_batch_request(
        self,
        *,
        batch_request: BatchSubmitRequest,
        schema_cls: Any,
    ) -> str:
        raise unexpected_error

    monkeypatch.setattr(
        QuickAgent,
        "_execute_batch_request",
        fake_execute_batch_request,
    )

    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.json", kind="json", text="{}", data={})
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)
    qa.loaded = loaded
    qa.model = cast(OpenAIChatModel, object())
    qa.model_spec = ModelSpec(base_url="http://base.invalid/v1", model_name="m")
    qa.toolset = RecordingToolset()
    qa.tool_ids = []
    qa.run_input = run_input
    qa.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
    qa._recorder._http_traffic_entries = [
        {
            "event": "request",
            "request": {
                "method": "POST",
                "url": "http://from-log/v1/chat/completions",
                "headers": {"Content-Type": "application/json"},
                "body": '{"test":1}',
            },
        }
    ]
    with pytest.raises(UnexpectedModelBehavior) as raised:
        await qa._run_text_step(step=step)
    assert raised.value.message == "Exceeded maximum retries (1) for output validation"
    assert qa._recorder.execution_log
    assert qa._recorder.execution_log[-1].call_site == "run_text_step"
    curl_command = qa._recorder.execution_log[-1].to_curl()
    assert "http://from-log/v1/chat/completions" in curl_command


@pytest.mark.anyio
async def test_http_hook_recorders_store_entries_on_quick_agent() -> None:
    qa = _make_quick_agent_for_test()
    qa.model_settings_json = None
    request = httpx.Request(
        method="POST",
        url="http://localhost:11434/v1/chat/completions",
        headers={"x-test-header": "abc"},
        content=b'{"messages":[{"role":"user","content":"hello"}]}',
    )
    response = httpx.Response(
        status_code=200,
        request=request,
        headers={"x-response-id": "resp-1"},
        content=b'{"id":"ok"}',
    )
    await qa._recorder._record_http_request(request)
    await qa._recorder._record_http_response(response)
    assert len(qa._recorder.http_request_log) == 1
    assert len(qa._recorder.http_response_log) == 1
    assert qa._recorder.http_request_log[0]["method"] == "POST"
    assert qa._recorder.http_response_log[0]["status_code"] == 200
    context = qa._recorder._last_http_exchange_context()
    assert context["request_source"] == "quick_agent_http_traffic_log"
    assert context["request"] == qa._recorder.http_request_log[-1]
    assert context["response"] == qa._recorder.http_response_log[-1]


def test_write_last_step_output_serializes_model(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    out_path = safe_root / "out.json"
    output = OutputSpec(file=str(out_path), format="json")
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step], output=output)

    permissions = DirectoryPermissions(safe_root)
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa.loaded = loaded
    qa.permissions = permissions
    result_path = write_output(
        qa.loaded.spec.output.file,
        OutputSchema(msg="hi"),
        permissions,
    )

    assert result_path == out_path
    assert '"msg": "hi"' in out_path.read_text(encoding="utf-8")


def test_write_last_step_output_writes_text(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    out_path = safe_root / "out.txt"
    output = OutputSpec(file=str(out_path), format="markdown")
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step], output=output)

    permissions = DirectoryPermissions(safe_root)
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa.loaded = loaded
    qa.permissions = permissions
    result_path = write_output(
        qa.loaded.spec.output.file,
        "hello",
        permissions,
    )

    assert result_path == out_path
    assert out_path.read_text(encoding="utf-8") == "hello"


@pytest.mark.anyio
async def test_handle_handoff_runs_followup() -> None:
    handoff = HandoffSpec(enabled=True, agent_id="next")
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step], handoff=handoff)

    qa = HandoffQuickAgent()
    qa.loaded = loaded
    await qa._handle_handoff("hello")

    assert len(qa.calls) == 1
    agent_id, input_data = qa.calls[0]
    assert agent_id == "next"
    assert isinstance(input_data, input_adaptors_module.TextInput)
    run_input = input_data.load()
    assert run_input.kind == "text"
    assert run_input.text == "hello"


@pytest.mark.anyio
async def test_handle_handoff_returns_followup_output() -> None:
    handoff = HandoffSpec(enabled=True, agent_id="next")
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step], handoff=handoff)

    qa = HandoffQuickAgent()
    qa.loaded = loaded
    output = await qa._handle_handoff("hello")

    assert output == "ok"


@pytest.mark.anyio
async def test_handle_handoff_serializes_structured_output() -> None:
    handoff = HandoffSpec(enabled=True, agent_id="next")
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step], handoff=handoff)

    qa = HandoffQuickAgent()
    qa.loaded = loaded
    await qa._handle_handoff(OutputSchema(msg="hi"))

    assert len(qa.calls) == 1
    _, input_data = qa.calls[0]
    assert isinstance(input_data, input_adaptors_module.TextInput)
    run_input = input_data.load()
    assert '"msg": "hi"' in run_input.text


@pytest.mark.anyio
async def test_handle_handoff_skips_when_disabled() -> None:
    handoff = HandoffSpec(enabled=False, agent_id="next")
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step], handoff=handoff)

    qa = HandoffQuickAgent()
    qa.loaded = loaded
    await qa._handle_handoff("ignored")

    assert qa.calls == []


@pytest.mark.anyio
async def test_run_agent_wires_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        tools=["tool.a", "agent_call", "tool.a"],
        output=OutputSpec(file=str(tmp_path / "out.json")),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )

    run_input = RunInput(
        source_path=str(tmp_path / "input.json"), kind="json", text="{}", data={}
    )
    toolset = RecordingToolset()
    model = object()
    settings = {"extra_body": {"format": "json"}}
    out_path = tmp_path / "out.json"

    load_input_recorder = SyncCallRecorder(return_value=run_input)
    build_model_recorder = SyncCallRecorder(return_value=model)
    build_toolset_recorder = SyncCallRecorder(return_value=toolset)
    build_settings_recorder = SyncCallRecorder(return_value=settings)
    maybe_inject_recorder = SyncCallRecorder(return_value=None)
    run_chain_recorder = AsyncCallRecorder(return_value={"result": "final"})
    write_output_recorder = SyncCallRecorder(return_value=out_path)
    handoff_recorder = AsyncCallRecorder(return_value=None)

    monkeypatch.setattr(input_adaptors_module, "load_input", load_input_recorder)
    monkeypatch.setattr(qa_module, "build_model", build_model_recorder)
    monkeypatch.setattr(QuickAgent, "_build_model_settings", build_settings_recorder)
    monkeypatch.setattr(QuickAgent, "_run_chain", run_chain_recorder)
    monkeypatch.setattr(QuickAgent, "_write_last_step_output", write_output_recorder)
    monkeypatch.setattr(QuickAgent, "_handle_handoff", handoff_recorder)

    tools = AgentTools([tmp_path])
    monkeypatch.setattr(tools, "build_toolset", build_toolset_recorder)
    monkeypatch.setattr(tools, "maybe_inject_agent_call", maybe_inject_recorder)
    fake_registry = FakeRegistry(loaded)

    agent = QuickAgent(
        registry=fake_registry,
        tools=tools,
        directory_permissions=_permissions(tmp_path),
        agent_id="agent-1",
        input_data=tmp_path / "input.json",
        extra_tools=["tool.b"],
    )

    result = await agent.run()

    assert result == {"result": "final"}
    assert fake_registry.calls == ["agent-1"]

    assert load_input_recorder.calls
    load_args, load_kwargs = load_input_recorder.calls[0]
    assert load_kwargs == {}
    assert load_args[0] == tmp_path / "input.json"
    assert isinstance(load_args[1], DirectoryPermissions)
    assert load_args[1].root == _permissions(tmp_path).root
    assert len(build_model_recorder.calls) == 1
    build_model_args, build_model_kwargs = build_model_recorder.calls[0]
    assert build_model_args == (loaded.spec.model,)
    assert "http_client" in build_model_kwargs

    assert build_settings_recorder.calls == [((loaded.spec.model,), {})]

    assert build_toolset_recorder.calls
    args, kwargs = build_toolset_recorder.calls[0]
    assert kwargs == {}
    assert args[0] == [
        "tool.a",
        "agent_call",
        "tool.b",
    ]
    assert isinstance(args[1], DirectoryPermissions)

    maybe_args, maybe_kwargs = maybe_inject_recorder.calls[0]
    assert maybe_kwargs == {}
    assert maybe_args[0] == [
        "tool.a",
        "agent_call",
        "tool.b",
    ]
    assert maybe_args[1] is toolset
    assert maybe_args[2] == run_input.source_path
    assert callable(maybe_args[3])

    assert run_chain_recorder.calls
    assert run_chain_recorder.calls[0]["kwargs"] == {}

    assert write_output_recorder.calls
    write_args, write_kwargs = write_output_recorder.calls[0]
    assert write_kwargs == {}
    assert write_args[0] == {"result": "final"}
    assert handoff_recorder.calls == [{"args": ({"result": "final"},), "kwargs": {}}]


@pytest.mark.anyio
async def test_run_skips_write_when_output_file_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )

    run_input = RunInput(
        source_path=str(tmp_path / "input.json"), kind="json", text="{}", data={}
    )
    toolset = RecordingToolset()
    model = object()

    load_input_recorder = SyncCallRecorder(return_value=run_input)
    build_model_recorder = SyncCallRecorder(return_value=model)
    build_toolset_recorder = SyncCallRecorder(return_value=toolset)
    build_settings_recorder = SyncCallRecorder(return_value=None)
    maybe_inject_recorder = SyncCallRecorder(return_value=None)
    run_chain_recorder = AsyncCallRecorder(return_value={"result": "final"})
    write_output_recorder = SyncCallRecorder(return_value=tmp_path / "out.json")
    handoff_recorder = AsyncCallRecorder(return_value=None)

    monkeypatch.setattr(input_adaptors_module, "load_input", load_input_recorder)
    monkeypatch.setattr(qa_module, "build_model", build_model_recorder)
    monkeypatch.setattr(QuickAgent, "_build_model_settings", build_settings_recorder)
    monkeypatch.setattr(QuickAgent, "_run_chain", run_chain_recorder)
    monkeypatch.setattr(QuickAgent, "_write_last_step_output", write_output_recorder)
    monkeypatch.setattr(QuickAgent, "_handle_handoff", handoff_recorder)

    tools = AgentTools([tmp_path])
    monkeypatch.setattr(tools, "build_toolset", build_toolset_recorder)
    monkeypatch.setattr(tools, "maybe_inject_agent_call", maybe_inject_recorder)
    fake_registry = FakeRegistry(loaded)

    agent = QuickAgent(
        registry=fake_registry,
        tools=tools,
        directory_permissions=_permissions(tmp_path),
        agent_id="agent-1",
        input_data=tmp_path / "input.json",
        extra_tools=None,
    )

    result = await agent.run()

    assert result == {"result": "final"}
    assert write_output_recorder.calls == []
    assert handoff_recorder.calls == [{"args": ({"result": "final"},), "kwargs": {}}]


def test_init_can_disable_http_traffic_recording(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    run_input = RunInput(
        source_path=str(tmp_path / "input.json"), kind="json", text="{}", data={}
    )
    load_input_recorder = SyncCallRecorder(return_value=run_input)
    build_model_recorder = SyncCallRecorder(return_value=object())
    monkeypatch.setattr(input_adaptors_module, "load_input", load_input_recorder)
    monkeypatch.setattr(qa_module, "build_model", build_model_recorder)
    tools = AgentTools([tmp_path])
    monkeypatch.setattr(
        tools, "build_toolset", SyncCallRecorder(return_value=RecordingToolset())
    )
    fake_registry = FakeRegistry(loaded)
    QuickAgent(
        registry=fake_registry,
        tools=tools,
        directory_permissions=_permissions(tmp_path),
        agent_id="agent-1",
        input_data=tmp_path / "input.json",
        extra_tools=None,
        record_http_traffic=False,
    )
    assert len(build_model_recorder.calls) == 1
    build_model_args, build_model_kwargs = build_model_recorder.calls[0]
    assert build_model_args == (loaded.spec.model,)
    http_client = build_model_kwargs.get("http_client")
    assert isinstance(http_client, httpx.AsyncClient)
    asyncio.run(http_client.aclose())


def test_init_http_traffic_recording_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    run_input = RunInput(
        source_path=str(tmp_path / "input.json"), kind="json", text="{}", data={}
    )
    load_input_recorder = SyncCallRecorder(return_value=run_input)
    build_model_recorder = SyncCallRecorder(return_value=object())
    monkeypatch.setattr(input_adaptors_module, "load_input", load_input_recorder)
    monkeypatch.setattr(qa_module, "build_model", build_model_recorder)
    tools = AgentTools([tmp_path])
    monkeypatch.setattr(
        tools, "build_toolset", SyncCallRecorder(return_value=RecordingToolset())
    )
    fake_registry = FakeRegistry(loaded)
    QuickAgent(
        registry=fake_registry,
        tools=tools,
        directory_permissions=_permissions(tmp_path),
        agent_id="agent-1",
        input_data=tmp_path / "input.json",
        extra_tools=None,
    )
    assert len(build_model_recorder.calls) == 1
    build_model_args, build_model_kwargs = build_model_recorder.calls[0]
    assert build_model_args == (loaded.spec.model,)
    http_client = build_model_kwargs.get("http_client")
    assert isinstance(http_client, httpx.AsyncClient)
    asyncio.run(http_client.aclose())


@pytest.mark.anyio
async def test_init_applies_model_http_client_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(
            base_url="http://x",
            model_name="m",
            timeout_seconds=321.0,
            keepalive_expiry_seconds=123.0,
        ),
        chain=[step],
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    run_input = RunInput(
        source_path=str(tmp_path / "input.json"), kind="json", text="{}", data={}
    )
    load_input_recorder = SyncCallRecorder(return_value=run_input)
    build_model_recorder = SyncCallRecorder(return_value=object())
    monkeypatch.setattr(input_adaptors_module, "load_input", load_input_recorder)
    monkeypatch.setattr(qa_module, "build_model", build_model_recorder)
    tools = AgentTools([tmp_path])
    monkeypatch.setattr(
        tools, "build_toolset", SyncCallRecorder(return_value=RecordingToolset())
    )
    fake_registry = FakeRegistry(loaded)
    QuickAgent(
        registry=fake_registry,
        tools=tools,
        directory_permissions=_permissions(tmp_path),
        agent_id="agent-1",
        input_data=tmp_path / "input.json",
        extra_tools=None,
    )
    assert len(build_model_recorder.calls) == 1
    _, build_model_kwargs = build_model_recorder.calls[0]
    http_client = build_model_kwargs["http_client"]
    assert isinstance(http_client, httpx.AsyncClient)
    assert http_client.timeout.read == 321.0
    await http_client.aclose()


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("nested_output", "expected_write_output"),
    [
        ("inline", False),
        ("file", True),
    ],
)
async def test_run_nested_agent_respects_nested_output(
    monkeypatch: pytest.MonkeyPatch,
    nested_output: Literal["inline", "file"],
    expected_write_output: bool,
) -> None:
    qa = _make_quick_agent_for_test()
    qa._registry = cast(AgentRegistry, object())
    qa._tools = cast(AgentTools, object())
    qa._directory_permissions = cast(DirectoryPermissions, object())
    qa.model_spec = ModelSpec(base_url="http://x", model_name="m")
    qa._recorder._enable_llm_request_logging = True
    qa._recorder._llm_log_path = Path("log/custom.log")

    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    loaded.spec.nested_output = nested_output
    qa.loaded = loaded

    init_recorder = SyncCallRecorder(return_value=None)
    run_recorder = AsyncCallRecorder(return_value="ok")
    monkeypatch.setattr(QuickAgent, "__init__", init_recorder)
    monkeypatch.setattr(QuickAgent, "run", run_recorder)

    await qa._run_nested_agent("child", Path("input.txt"))

    assert len(init_recorder.calls) == 1
    _, kwargs = init_recorder.calls[0]
    assert kwargs["write_output"] is expected_write_output
    assert kwargs["enable_llm_request_logging"] is True
    assert kwargs["llm_log_path"] == Path("log/custom.log")


@pytest.mark.anyio
async def test_orchestrator_batch_uses_same_arguments_as_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch = Orchestrator([Path("agents")], [Path("tools")], safe_dir=Path("safe"))
    run_input = input_adaptors_module.TextInput("hello")
    init_recorder = SyncCallRecorder(return_value=None)
    batch_recorder = SyncCallRecorder(
        return_value=BatchSubmitRequest(
            request_id="r1",
            agent_id="file-manager",
            step_id=None,
            step_kind="single_shot",
            model=BatchModelConfig(
                provider="openai-compatible", base_url="http://x", model_name="m"
            ),
            messages=[BatchMessage(role="user", content="hello")],
        )
    )
    monkeypatch.setattr(QuickAgent, "__init__", init_recorder)
    monkeypatch.setattr(QuickAgent, "batch", batch_recorder)
    request = await orch.batch(
        "file-manager",
        run_input,
        record_http_traffic=False,
        enable_llm_request_logging=False,
    )
    assert request.agent_id == "file-manager"


@pytest.mark.anyio
async def test_orchestrator_import_uses_same_arguments_as_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orch = Orchestrator([Path("agents")], [Path("tools")], safe_dir=Path("safe"))
    run_input = input_adaptors_module.TextInput("hello")
    init_recorder = SyncCallRecorder(return_value=None)
    import_recorder = AsyncCallRecorder(return_value=BatchImportOutcome(result="ok"))
    monkeypatch.setattr(QuickAgent, "__init__", init_recorder)
    monkeypatch.setattr(QuickAgent, "import_result", import_recorder)
    request = BatchImportRequest(
        request_id="r1",
        payload={"state": "completed", "output": "done"},
    )
    outcome = await orch.import_result(
        "file-manager",
        run_input,
        request,
        record_http_traffic=False,
        enable_llm_request_logging=False,
    )
    assert outcome.result == "ok"


def test_create_batch_request_for_current_step() -> None:
    qa = _make_quick_agent_for_test()
    request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="do thing",
        system_prompt="system prompt",
        user_prompt="input prompt",
        model_settings=qa.model_settings_json,
    )
    assert request.agent_id == "a"
    assert request.step_id == "s1"
    assert request.step_kind == "text"
    assert request.model.model_name == qa.model_spec.model_name
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert request.messages[1].content == "input prompt"


def test_batch_submit_jsonl_line_uses_open_weight_invoke_model_input_shape() -> None:
    request = BatchSubmitRequest(
        request_id="r1",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="http://x",
            model_name="m",
            temperature=0.1,
            max_completion_tokens=256,
            extra_body={
                "inferenceConfig": {"topP": 0.9, "max_new_tokens": 300},
                "response_format": {"type": "json_schema"},
                "requestMetadata": {"k": "v"},
            },
        ),
        messages=[
            BatchMessage(role="system", content="sys"),
            BatchMessage(role="user", content="hello"),
        ],
    )
    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    assert model_input["response_format"] == {"type": "json_schema"}
    messages = model_input["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {
        "role": "system",
        "content": "sys",
    }
    assert messages[1] == {
        "role": "user",
        "content": "hello",
    }
    assert "system" not in model_input
    inference_config = model_input["inferenceConfig"]
    assert isinstance(inference_config, dict)
    assert inference_config["maxTokens"] == 300
    assert inference_config["temperature"] == 0.1
    assert inference_config["topP"] == 0.9


def test_batch_submit_jsonl_line_uses_converse_model_input_shape() -> None:
    request = BatchSubmitRequest(
        request_id="r2",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="http://x",
            model_name="m",
            temperature=0.1,
            max_completion_tokens=256,
            bedrock_request_mode="converse",
        ),
        messages=[
            BatchMessage(role="system", content="sys"),
            BatchMessage(role="user", content="hello"),
        ],
    )
    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    assert model_input["system"] == [{"type": "text", "text": "sys"}]
    messages = model_input["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {
        "role": "user",
        "content": [{"type": "text", "text": "hello"}],
    }


def test_batch_submit_jsonl_line_uses_anthropic_invoke_model_input_shape() -> None:
    request = BatchSubmitRequest(
        request_id="r3",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="http://x",
            model_name="m",
            bedrock_request_mode="anthropic_invoke",
            extra_body={"anthropic_version": "bedrock-2023-05-31"},
        ),
        response_format={"type": "json_schema"},
        messages=[
            BatchMessage(role="system", content="sys"),
            BatchMessage(role="user", content="hello"),
        ],
    )
    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    assert model_input["anthropic_version"] == "bedrock-2023-05-31"
    assert "response_format" not in model_input
    output_config = model_input["output_config"]
    assert isinstance(output_config, dict)
    assert output_config["format"] == {"type": "json_schema"}


def test_batch_submit_jsonl_line_inferrs_anthropic_mode_from_model_name() -> None:
    request = BatchSubmitRequest(
        request_id="r4",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="http://x",
            model_name="anthropic.claude-3-7-sonnet-20250219-v1:0",
        ),
        response_format={"type": "json_schema"},
        messages=[BatchMessage(role="user", content="hello")],
    )
    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    assert "output_config" in model_input
    assert "response_format" not in model_input


def test_batch_submit_jsonl_line_inferrs_qwen_mode_from_model_name() -> None:
    request = BatchSubmitRequest(
        request_id="r5",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="http://x",
            model_name="qwen.qwen3-next-80b-a3b",
        ),
        response_format={"type": "json_schema"},
        messages=[BatchMessage(role="user", content="hello")],
    )
    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    assert "response_format" in model_input
    assert "output_config" not in model_input


def test_batch_entry_point_returns_submit_request() -> None:
    qa = _make_quick_agent_for_test()
    request = qa.batch()
    assert request.step_id == "s1"
    assert request.step_kind == "text"


def test_batch_structured_step_includes_response_format_for_non_openai() -> None:
    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Out"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://localhost:11434/v1", model_name="m"),
        chain=[step],
        schemas={"Out": f"{__name__}:ExampleSchema"},
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    qa = _make_quick_agent_for_test(loaded=loaded)
    request = qa.batch()
    assert request.response_format is not None
    assert request.response_format["type"] == "json_schema"


@pytest.mark.anyio
async def test_apply_imported_batch_result_returns_next_request() -> None:
    qa = _make_quick_agent_for_test()
    next_request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="do thing",
        system_prompt="system prompt",
        user_prompt="input prompt",
        model_settings=qa.model_settings_json,
    )
    batch_import = BatchImportRequest(
        request_id="r1",
        payload={
            "state": "submit_next",
            "next_request": next_request.model_dump(mode="json"),
        },
    )
    outcome = await qa.import_result(batch_import=batch_import)
    assert outcome.next_request is not None
    assert outcome.next_request.request_id == next_request.request_id


@pytest.mark.anyio
async def test_import_entry_point_returns_outcome() -> None:
    qa = _make_quick_agent_for_test()
    qa.loaded.spec.output.format = "markdown"
    batch_import = BatchImportRequest(
        request_id="r1",
        payload={"state": "completed", "output": "ok"},
    )
    outcome = await qa.import_result(batch_import=batch_import)
    assert outcome.result == "ok"


@pytest.mark.anyio
async def test_import_entry_point_allows_single_shot_dict_without_schema() -> None:
    qa = _make_quick_agent_for_test()
    qa.loaded.spec.output.output_schema = None
    qa.loaded.spec.chain = []
    batch_import = BatchImportRequest(
        request_id="r1",
        payload={"state": "completed", "output": {"foo": "bar"}},
    )
    outcome = await qa.import_result(batch_import=batch_import)
    assert outcome.result == {"foo": "bar"}


@pytest.mark.anyio
async def test_apply_imported_batch_result_maps_tools_not_supported_error() -> None:
    qa = _make_quick_agent_for_test()
    batch_import = BatchImportRequest(
        request_id="r1",
        payload={"state": "error", "message": "model does not support tools"},
    )
    with pytest.raises(QuickAgentToolsNotSupportedException):
        await qa.import_result(batch_import=batch_import)


@pytest.mark.anyio
async def test_import_chain_result_accepts_dict_for_structured_step() -> None:
    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Out"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        schemas={"Out": f"{__name__}:ExampleSchema"},
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    qa = _make_quick_agent_for_test(loaded=loaded)
    batch_import = BatchImportRequest(
        request_id="r1",
        payload={"state": "completed", "output": {"x": 5}},
    )
    outcome = await qa.import_result(batch_import=batch_import)
    assert outcome.result == ExampleSchema(x=5)
    assert isinstance(qa.state["steps"]["s1"], ExampleSchema)
    assert isinstance(qa.state["last_step_output"], ExampleSchema)


def test_parse_structured_result_rejects_wrong_basemodel_type() -> None:
    qa = _make_quick_agent_for_test()
    with pytest.raises(ValidationError):
        qa._parse_structured_result(OtherSchema(msg="x"), ExampleSchema)


@pytest.mark.anyio
async def test_run_text_step_uses_batch_call_handler() -> None:
    async def batch_call(request: BatchSubmitRequest) -> BatchImportRequest:
        return BatchImportRequest(
            request_id=request.request_id,
            payload={"state": "completed", "output": "from batch"},
        )

    qa = _make_quick_agent_for_test(memory={"batch_call": batch_call})
    step = qa.loaded.spec.chain[0]
    output = await qa._run_text_step(step=step)
    assert output == "from batch"


@pytest.mark.anyio
async def test_run_structured_step_parses_batch_output_with_json_fallback() -> None:
    async def batch_call(request: BatchSubmitRequest) -> BatchImportRequest:
        return BatchImportRequest(
            request_id=request.request_id,
            payload={"state": "completed", "output": 'prefix {"x": 7} suffix'},
        )

    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Out"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        schemas={"Out": f"{__name__}:ExampleSchema"},
        output=OutputSpec(file=None),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    qa = _make_quick_agent_for_test(loaded=loaded, memory={"batch_call": batch_call})
    output = await qa._run_structured_step(step=step)
    assert isinstance(output, ExampleSchema)
    assert output.x == 7


def test_import_outcome_handles_tool_use_state() -> None:
    qa = _make_quick_agent_for_test()
    submit_request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="do thing",
        system_prompt="system",
        user_prompt="user prompt",
        model_settings=None,
    )
    batch_import = BatchImportRequest(
        request_id="r1",
        payload={
            "state": "tool_use",
            "tool_calls": [{"id": "call1", "name": "tool_name", "arguments": {"x": 1}}],
            "submit_request": submit_request.model_dump(mode="json"),
        },
    )
    outcome = qa._import_outcome(batch_import=batch_import)
    assert outcome.tool_calls is not None
    assert len(outcome.tool_calls) == 1
    assert outcome.tool_calls[0]["id"] == "call1"
    assert outcome.pending_submit_request is not None


def test_import_outcome_tool_use_missing_tool_calls_raises() -> None:
    qa = _make_quick_agent_for_test()
    batch_import = BatchImportRequest(
        request_id="r1",
        payload={"state": "tool_use"},
    )
    with pytest.raises(ValueError, match="tool_calls"):
        qa._import_outcome(batch_import=batch_import)


def test_build_next_request_with_tool_results_extends_messages() -> None:
    qa = _make_quick_agent_for_test()
    submit_request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="do thing",
        system_prompt="system",
        user_prompt="user prompt",
        model_settings=None,
    )
    tool_calls = [{"id": "call1", "name": "tool_name", "arguments": {"x": 1}}]
    executed = [ToolCallResult(id="call1", name="tool_name", content="result text")]
    next_req = qa._build_next_request_with_tool_results(
        tool_calls=tool_calls,
        executed=executed,
        submit_request=submit_request,
    )
    assert len(next_req.messages) == len(submit_request.messages) + 2
    assistant_msg = next_req.messages[-2]
    assert assistant_msg.role == "assistant"
    assert assistant_msg.tool_calls is not None
    tool_msg = next_req.messages[-1]
    assert tool_msg.role == "tool"
    assert tool_msg.content == "result text"
    assert tool_msg.tool_call_id == "call1"


def test_build_converse_jsonl_line_with_tool_calls() -> None:
    request = BatchSubmitRequest(
        request_id="r1",
        agent_id="a",
        step_id=None,
        step_kind="text",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="http://x",
            model_name="m",
            bedrock_request_mode="converse",
        ),
        messages=[
            BatchMessage(role="system", content="sys"),
            BatchMessage(role="user", content="hello"),
            BatchMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {"name": "my_tool", "arguments": '{"x": 1}'},
                    }
                ],
            ),
            BatchMessage(
                role="tool", content="result", tool_call_id="call1", name="my_tool"
            ),
        ],
    )
    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    messages = model_input["messages"]
    assert len(messages) == 3  # user, assistant, user(tool_result)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assistant_content = messages[1]["content"]
    assert any("toolUse" in block for block in assistant_content)
    assert messages[2]["role"] == "user"
    tool_result_content = messages[2]["content"]
    assert any("toolResult" in block for block in tool_result_content)


def test_batch_message_supports_tool_calls_field() -> None:
    msg = BatchMessage(
        role="assistant",
        tool_calls=[
            {
                "id": "x",
                "type": "function",
                "function": {"name": "t", "arguments": "{}"},
            }
        ],
    )
    assert msg.tool_calls is not None
    assert msg.content is None


def test_batch_submit_request_supports_tools_field() -> None:
    req = BatchSubmitRequest(
        request_id="r1",
        agent_id="a",
        step_id=None,
        step_kind="text",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="http://x",
            model_name="m",
        ),
        messages=[BatchMessage(role="user", content="hello")],
        tools=[
            BatchToolDefinition(
                name="t", description="test tool", input_schema={"type": "object"}
            )
        ],
        tool_use_enabled=True,
    )
    assert req.tool_use_enabled is True
    assert req.tools is not None
    assert req.tools[0].name == "t"
