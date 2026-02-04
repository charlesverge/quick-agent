import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset

from quick_agent import quick_agent as qa_module
from quick_agent import agent_tools as tools_module
from quick_agent import input_adaptors as input_adaptors_module
from quick_agent.agent_call_tool import AgentCallTool
from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.models import AgentSpec
from quick_agent.models import ChainStepSpec
from quick_agent.models import LoadedAgentFile
from quick_agent.models import ModelSpec
from quick_agent.models.handoff_spec import HandoffSpec
from quick_agent.models.output_spec import OutputSpec
from quick_agent.models.run_input import RunInput
from quick_agent.orchestrator import Orchestrator
from quick_agent.quick_agent import QuickAgent
from quick_agent.quick_agent import build_model
from quick_agent.quick_agent import resolve_schema


class DummyProvider:
    def __init__(self, base_url: str, api_key: str) -> None:
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

    def add_function(self, *args: Any, **kwargs: Any) -> None:
        func = kwargs.get("func")
        name = kwargs.get("name")
        description = kwargs.get("description")
        if func is not None and name is not None and description is not None:
            self.add_calls.append((func, name, description))


class FakeAgentResult:
    def __init__(self, output: str) -> None:
        self.output = output


class FakeAgent:
    next_output = ""
    last_init: dict[str, Any] | None = None
    last_prompt: str | None = None

    def __init__(
        self,
        model: Any,
        instructions: str,
        toolsets: list[Any],
        output_type: Any,
        model_settings: Any | None = None,
    ) -> None:
        FakeAgent.last_init = {
            "model": model,
            "instructions": instructions,
            "toolsets": toolsets,
            "output_type": output_type,
            "model_settings": model_settings,
        }

    async def run(self, user_prompt: str) -> FakeAgentResult:
        FakeAgent.last_prompt = user_prompt
        return FakeAgentResult(FakeAgent.next_output)


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

    async def _run_step(self, **kwargs: Any) -> tuple[Any, Any]:
        step = kwargs.get("step")
        if step is not None:
            self.calls.append(step.id)
        output = self.outputs[self.index]
        self.index += 1
        return output


class HandoffQuickAgent(QuickAgent):
    def __init__(self) -> None:
        self.calls: list[tuple[str, Path]] = []

    async def _run_nested_agent(self, agent_id: str, input_path: Path) -> str:
        self.calls.append((agent_id, input_path))
        return "ok"


class ExampleSchema(BaseModel):
    x: int


class OutputSchema(BaseModel):
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
    return LoadedAgentFile(spec=spec, body="system", step_prompts={"step:one": "do thing"})


def _permissions(tmp_path: Path | None = None) -> DirectoryPermissions:
    root = Path("safe") if tmp_path is None else tmp_path / "safe"
    return DirectoryPermissions(root)


def test_init_sets_registry_and_tool_roots(tmp_path: Path) -> None:
    orch = Orchestrator([tmp_path], [tmp_path / "tools"], safe_dir=_permissions(tmp_path).root)

    assert isinstance(orch.registry, AgentRegistry)
    assert isinstance(orch.tools, AgentTools)
    assert orch.tools._tool_roots == [tmp_path / "tools"]


def test_build_model_uses_env_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "abc")
    monkeypatch.setattr(qa_module, "OpenAIProvider", DummyProvider)
    monkeypatch.setattr(qa_module, "OpenAIChatModel", DummyModel)

    spec = ModelSpec(base_url="http://base", model_name="gpt-test", api_key_env="TEST_KEY")
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
    loaded = LoadedAgentFile(spec=spec, body="", step_prompts={})

    try:
        assert resolve_schema(loaded, "Good") is GoodSchema
        with pytest.raises(KeyError):
            resolve_schema(loaded, "Missing")
        with pytest.raises(TypeError):
            resolve_schema(loaded, "Bad")
    finally:
        sys.modules.pop("schemas.orch", None)


def test_build_toolset_filters_agent_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sentinel_toolset = RecordingToolset()
    recorder = LoadToolsRecorder(sentinel_toolset)
    monkeypatch.setattr(tools_module, "load_tools", recorder)
    tools = AgentTools([tmp_path])
    toolset = tools.build_toolset(["tool.a", "agent.call", "tool.b"], _permissions(tmp_path))

    assert toolset is sentinel_toolset
    assert len(recorder.calls) == 1
    roots, tool_ids, perms = recorder.calls[0]
    assert roots == [tmp_path]
    assert tool_ids == ["tool.a", "tool.b"]
    assert perms.root == _permissions(tmp_path).root


def test_build_toolset_returns_empty_for_agent_call_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    recorder = LoadToolsRecorder(RecordingToolset())
    monkeypatch.setattr(tools_module, "load_tools", recorder)
    tools = AgentTools([tmp_path])
    toolset = tools.build_toolset(["agent.call"], _permissions(tmp_path))

    assert isinstance(toolset, FunctionToolset)
    assert recorder.calls == []


def test_maybe_inject_agent_call_tool_adds_when_requested() -> None:
    tools = AgentTools([])
    toolset = RecordingToolset()

    tools.maybe_inject_agent_call(
        ["agent.call"],
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


def test_init_state_contains_agent_id_and_steps() -> None:
    qa = object.__new__(QuickAgent)

    state = qa._init_state("agent-1")

    assert state == {"agent_id": "agent-1", "steps": {}}


def test_build_model_settings_openai_compatible() -> None:
    qa = object.__new__(QuickAgent)
    spec = ModelSpec(base_url="http://x", model_name="m", provider="openai-compatible")

    settings = qa._build_model_settings(spec)

    assert settings == {"extra_body": {"format": "json"}}


def test_build_model_settings_openai_endpoint_skips_format() -> None:
    qa = object.__new__(QuickAgent)
    spec = ModelSpec(
        base_url="https://api.openai.com/v1",
        model_name="m",
        provider="openai-compatible",
    )

    settings = qa._build_model_settings(spec)

    assert settings is None


def test_build_model_settings_other_provider() -> None:
    qa = object.__new__(QuickAgent)
    spec = ModelSpec(base_url="http://x", model_name="m", provider="other")

    settings = qa._build_model_settings(spec)

    assert settings is None


def test_build_structured_model_settings_non_openai_passthrough() -> None:
    qa = object.__new__(QuickAgent)
    schema = ExampleSchema
    model = cast(OpenAIChatModel, DummyOpenAIModel("http://localhost"))
    settings: ModelSettings = {"extra_body": {"format": "json"}}

    result = qa._build_structured_model_settings(
        model=model,
        model_settings_json=settings,
        schema_cls=schema,
    )

    assert result == settings


def test_build_structured_model_settings_openai_injects_schema() -> None:
    qa = object.__new__(QuickAgent)
    schema = ExampleSchema
    model = cast(OpenAIChatModel, DummyOpenAIModel("https://api.openai.com/v1"))

    result = qa._build_structured_model_settings(
        model=model,
        model_settings_json=None,
        schema_cls=schema,
    )

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


def test_build_user_prompt_raises_for_missing_section() -> None:
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:missing")
    loaded = LoadedAgentFile(spec=_make_loaded_with_chain([step]).spec, body="body", step_prompts={})
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = object.__new__(QuickAgent)
    with pytest.raises(KeyError):
        qa._build_user_prompt(step=step, loaded=loaded, run_input=run_input, state={"steps": {}})


def test_build_user_prompt_uses_prompting(monkeypatch: pytest.MonkeyPatch) -> None:
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)
    recorder = SyncCallRecorder(return_value="prompt")
    monkeypatch.setattr(qa_module, "make_user_prompt", recorder)

    qa = object.__new__(QuickAgent)
    result = qa._build_user_prompt(step=step, loaded=loaded, run_input=run_input, state={"steps": {}})

    assert result == "prompt"
    assert recorder.calls == [((loaded.step_prompts["step:one"], run_input, {"steps": {}}), {})]


@pytest.mark.anyio
async def test_run_step_text_returns_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa_module, "Agent", FakeAgent)
    FakeAgent.next_output = "hello"

    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = object.__new__(QuickAgent)
    output, final = await qa._run_step(
        step=step,
        loaded=loaded,
        model=cast(OpenAIChatModel, object()),
        model_settings_json=None,
        toolset=RecordingToolset(),
        run_input=run_input,
        state={"agent_id": "a", "steps": {}},
    )

    assert output == "hello"
    assert final == "hello"
    assert FakeAgent.last_init is not None
    assert FakeAgent.last_init["instructions"] == "system"
    assert FakeAgent.last_init["output_type"] is str


@pytest.mark.anyio
async def test_run_step_structured_parses_json_with_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa_module, "Agent", FakeAgent)
    FakeAgent.next_output = "preface {\"x\": 7} trailing"

    schema_module = types.ModuleType("schemas.struct")
    schema_module.__dict__["ExampleSchema"] = ExampleSchema
    sys.modules["schemas.struct"] = schema_module

    step = ChainStepSpec(id="s1", kind="structured", prompt_section="step:one", output_schema="Example")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        schemas={"Example": "schemas.struct:ExampleSchema"},
    )
    loaded = LoadedAgentFile(spec=spec, body="system", step_prompts={"step:one": "do thing"})
    run_input = RunInput(source_path="in.json", kind="json", text="{}", data={})

    try:
        qa = object.__new__(QuickAgent)
        output, final = await qa._run_step(
            step=step,
            loaded=loaded,
            model=cast(OpenAIChatModel, object()),
            model_settings_json={"extra_body": {"format": "json"}},
            toolset=RecordingToolset(),
            run_input=run_input,
            state={"agent_id": "a", "steps": {}},
        )
    finally:
        sys.modules.pop("schemas.struct", None)

    assert output == {"x": 7}
    assert isinstance(final, ExampleSchema)
    assert final.x == 7


@pytest.mark.anyio
async def test_run_step_unknown_kind_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa_module, "Agent", FakeAgent)
    step = ChainStepSpec(id="s1", kind="mystery", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = object.__new__(QuickAgent)
    with pytest.raises(NotImplementedError):
        await qa._run_step(
            step=step,
            loaded=loaded,
            model=cast(OpenAIChatModel, object()),
            model_settings_json=None,
            toolset=RecordingToolset(),
            run_input=run_input,
            state={"agent_id": "a", "steps": {}},
        )


@pytest.mark.anyio
async def test_run_text_step_uses_build_user_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa_module, "Agent", FakeAgent)
    FakeAgent.next_output = "ok"
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.txt", kind="text", text="hi", data=None)

    qa = object.__new__(QuickAgent)
    monkeypatch.setattr(qa, "_build_user_prompt", SyncCallRecorder(return_value="prompt"))

    output, final = await qa._run_text_step(
        step=step,
        loaded=loaded,
        model=cast(OpenAIChatModel, object()),
        toolset=RecordingToolset(),
        run_input=run_input,
        state={"agent_id": "a", "steps": {}},
    )

    assert output == "ok"
    assert final == "ok"
    assert FakeAgent.last_prompt == "prompt"


@pytest.mark.anyio
async def test_run_structured_step_missing_schema_raises() -> None:
    step = ChainStepSpec(id="s1", kind="structured", prompt_section="step:one", output_schema=None)
    loaded = _make_loaded_with_chain([step])
    run_input = RunInput(source_path="in.json", kind="json", text="{}", data={})

    qa = object.__new__(QuickAgent)
    with pytest.raises(ValueError):
        await qa._run_structured_step(
            step=step,
            loaded=loaded,
            model=cast(OpenAIChatModel, object()),
            model_settings_json=None,
            toolset=RecordingToolset(),
            run_input=run_input,
            state={"agent_id": "a", "steps": {}},
        )


@pytest.mark.anyio
async def test_run_structured_step_parses_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa_module, "Agent", FakeAgent)
    FakeAgent.next_output = "{\"x\": 3}"

    schema_module = types.ModuleType("schemas.struct2")
    schema_module.__dict__["ExampleSchema"] = ExampleSchema
    sys.modules["schemas.struct2"] = schema_module

    step = ChainStepSpec(id="s1", kind="structured", prompt_section="step:one", output_schema="Example")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        schemas={"Example": "schemas.struct2:ExampleSchema"},
    )
    loaded = LoadedAgentFile(spec=spec, body="system", step_prompts={"step:one": "do thing"})
    run_input = RunInput(source_path="in.json", kind="json", text="{}", data={})

    try:
        qa = object.__new__(QuickAgent)
        output, final = await qa._run_structured_step(
            step=step,
            loaded=loaded,
            model=cast(OpenAIChatModel, object()),
            model_settings_json=None,
            toolset=RecordingToolset(),
            run_input=run_input,
            state={"agent_id": "a", "steps": {}},
        )
    finally:
        sys.modules.pop("schemas.struct2", None)

    assert output == {"x": 3}
    assert isinstance(final, ExampleSchema)


@pytest.mark.anyio
async def test_run_structured_step_adds_json_schema_for_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qa_module, "Agent", FakeAgent)
    FakeAgent.next_output = "{\"x\": 9}"

    schema_module = types.ModuleType("schemas.struct3")
    schema_module.__dict__["ExampleSchema"] = ExampleSchema
    sys.modules["schemas.struct3"] = schema_module

    step = ChainStepSpec(id="s1", kind="structured", prompt_section="step:one", output_schema="Example")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="https://api.openai.com/v1", model_name="m"),
        chain=[step],
        schemas={"Example": "schemas.struct3:ExampleSchema"},
    )
    loaded = LoadedAgentFile(spec=spec, body="system", step_prompts={"step:one": "do thing"})
    run_input = RunInput(source_path="in.json", kind="json", text="{}", data={})

    try:
        qa = object.__new__(QuickAgent)
        await qa._run_structured_step(
            step=step,
            loaded=loaded,
            model=cast(OpenAIChatModel, DummyOpenAIModel("https://api.openai.com/v1")),
            model_settings_json=None,
            toolset=RecordingToolset(),
            run_input=run_input,
            state={"agent_id": "a", "steps": {}},
        )
    finally:
        sys.modules.pop("schemas.struct3", None)

    assert FakeAgent.last_init is not None
    settings = FakeAgent.last_init["model_settings"]
    assert isinstance(settings, dict)
    extra_body = settings["extra_body"]
    assert extra_body["response_format"]["type"] == "json_schema"
    assert extra_body["response_format"]["json_schema"]["name"] == "ExampleSchema"
    assert extra_body["response_format"]["json_schema"]["strict"] is True


@pytest.mark.anyio
async def test_run_chain_updates_state_and_returns_last() -> None:
    step1 = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    step2 = ChainStepSpec(id="s2", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step1, step2])

    qa = RecordingQuickAgent(outputs=[({"a": 1}, "first"), ("b", "second")])
    state = {"agent_id": "a", "steps": {}}

    final = await qa._run_chain(
        loaded=loaded,
        model=cast(OpenAIChatModel, object()),
        model_settings_json=None,
        toolset=RecordingToolset(),
        run_input=RunInput(source_path="in.txt", kind="text", text="hi", data=None),
        state=state,
    )

    assert final == "second"
    assert state["steps"] == {"s1": {"a": 1}, "s2": "b"}
    assert qa.calls == ["s1", "s2"]


def test_write_final_output_serializes_model(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    out_path = safe_root / "out.json"
    output = OutputSpec(file=str(out_path), format="json")
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step], output=output)

    permissions = DirectoryPermissions(safe_root)
    qa = object.__new__(QuickAgent)
    result_path = qa._write_final_output(loaded, OutputSchema(msg="hi"), permissions)

    assert result_path == out_path
    assert "\"msg\": \"hi\"" in out_path.read_text(encoding="utf-8")


def test_write_final_output_writes_text(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    out_path = safe_root / "out.txt"
    output = OutputSpec(file=str(out_path), format="markdown")
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step], output=output)

    permissions = DirectoryPermissions(safe_root)
    qa = object.__new__(QuickAgent)
    result_path = qa._write_final_output(loaded, "hello", permissions)

    assert result_path == out_path
    assert out_path.read_text(encoding="utf-8") == "hello"


@pytest.mark.anyio
async def test_handle_handoff_runs_followup() -> None:
    out_path = Path("/tmp/out.json")
    handoff = HandoffSpec(enabled=True, agent_id="next")
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step], handoff=handoff)

    qa = HandoffQuickAgent()
    await qa._handle_handoff(loaded, out_path)

    assert qa.calls == [("next", out_path)]


@pytest.mark.anyio
async def test_handle_handoff_skips_when_disabled() -> None:
    out_path = Path("/tmp/out.json")
    handoff = HandoffSpec(enabled=False, agent_id="next")
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    loaded = _make_loaded_with_chain([step], handoff=handoff)

    qa = HandoffQuickAgent()
    await qa._handle_handoff(loaded, out_path)

    assert qa.calls == []


@pytest.mark.anyio
async def test_run_agent_wires_dependencies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    step = ChainStepSpec(id="s1", kind="text", prompt_section="step:one")
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        tools=["tool.a", "agent.call", "tool.a"],
        output=OutputSpec(file=str(tmp_path / "out.json")),
    )
    loaded = LoadedAgentFile(spec=spec, body="system", step_prompts={"step:one": "do thing"})

    run_input = RunInput(source_path=str(tmp_path / "input.json"), kind="json", text="{}", data={})
    toolset = RecordingToolset()
    model = object()
    settings = {"extra_body": {"format": "json"}}
    out_path = tmp_path / "out.json"

    load_input_recorder = SyncCallRecorder(return_value=run_input)
    build_model_recorder = SyncCallRecorder(return_value=model)
    build_toolset_recorder = SyncCallRecorder(return_value=toolset)
    build_settings_recorder = SyncCallRecorder(return_value=settings)
    maybe_inject_recorder = SyncCallRecorder(return_value=None)
    run_chain_recorder = AsyncCallRecorder(return_value="final")
    write_output_recorder = SyncCallRecorder(return_value=out_path)
    handoff_recorder = AsyncCallRecorder(return_value=None)

    monkeypatch.setattr(input_adaptors_module, "load_input", load_input_recorder)
    monkeypatch.setattr(qa_module, "build_model", build_model_recorder)
    monkeypatch.setattr(QuickAgent, "_build_model_settings", build_settings_recorder)
    monkeypatch.setattr(QuickAgent, "_run_chain", run_chain_recorder)
    monkeypatch.setattr(QuickAgent, "_write_final_output", write_output_recorder)
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

    assert result == "final"
    assert fake_registry.calls == ["agent-1"]

    assert load_input_recorder.calls
    load_args, load_kwargs = load_input_recorder.calls[0]
    assert load_kwargs == {}
    assert load_args[0] == tmp_path / "input.json"
    assert isinstance(load_args[1], DirectoryPermissions)
    assert load_args[1].root == _permissions(tmp_path).root
    assert build_model_recorder.calls == [((loaded.spec.model,), {})]

    assert build_toolset_recorder.calls
    args, kwargs = build_toolset_recorder.calls[0]
    assert kwargs == {}
    assert args[0] == [
        "tool.a",
        "agent.call",
        "tool.b",
    ]
    assert isinstance(args[1], DirectoryPermissions)

    assert build_settings_recorder.calls == [((loaded.spec.model,), {})]
    maybe_args, maybe_kwargs = maybe_inject_recorder.calls[0]
    assert maybe_kwargs == {}
    assert maybe_args[0] == [
        "tool.a",
        "agent.call",
        "tool.b",
    ]
    assert maybe_args[1] is toolset
    assert maybe_args[2] == run_input.source_path
    assert callable(maybe_args[3])

    assert run_chain_recorder.calls
    run_chain_kwargs = run_chain_recorder.calls[0]["kwargs"]
    assert run_chain_kwargs["loaded"] is loaded
    assert run_chain_kwargs["model"] is model
    assert run_chain_kwargs["model_settings_json"] is settings
    assert run_chain_kwargs["toolset"] is toolset
    assert run_chain_kwargs["run_input"] is run_input
    assert run_chain_kwargs["state"]["agent_id"] == "agent-1"

    assert write_output_recorder.calls
    write_args, write_kwargs = write_output_recorder.calls[0]
    assert write_kwargs == {}
    assert write_args[0] is loaded
    assert write_args[1] == "final"
    assert isinstance(write_args[2], DirectoryPermissions)
    assert write_args[2].root == _permissions(tmp_path).root
    assert handoff_recorder.calls == [{"args": (loaded, out_path), "kwargs": {}}]
