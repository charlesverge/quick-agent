from pathlib import Path

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import TextInput
from quick_agent.models import AgentSpec, ChainStepSpec, LoadedAgentFile, ModelSpec
from quick_agent.models.batch_request import (
    BatchMessage,
    BatchModelConfig,
    BatchSubmitRequest,
    BatchToolDefinition,
)
from quick_agent.models.chain_step_spec import ToolChoice
from quick_agent.models.output_spec import OutputSpec
from quick_agent.quick_agent import QuickAgent


class StaticRegistry(AgentRegistry):
    def __init__(self, loaded: LoadedAgentFile) -> None:
        super().__init__([])
        self._loaded = loaded

    def get(self, agent_id: str) -> LoadedAgentFile:
        return self._loaded


def _make_agent(
    *,
    spec: AgentSpec,
    step_prompts: dict[str, str] | None = None,
    tools_root: Path,
    tmp_path: Path,
) -> QuickAgent:
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="",
        system_prompt="",
        step_prompts=step_prompts or {},
    )
    return QuickAgent(
        registry=StaticRegistry(loaded),
        tools=AgentTools([tools_root]),
        directory_permissions=DirectoryPermissions(tmp_path),
        agent_id="agent-1",
        input_data=TextInput("hello"),
        write_output=False,
    )


def test_tool_choice_string_mode_parses() -> None:
    choice = ToolChoice.model_validate("required")
    assert choice.mode == "required"


def test_tool_choice_agent_level_shorthand_parses() -> None:
    spec = AgentSpec(
        name="test",
        output=OutputSpec(file=None),
        tool_choice=ToolChoice.model_validate("required"),
    )
    assert spec.tool_choice is not None
    assert spec.tool_choice.mode == "required"


def test_tool_choice_chain_overrides_agent_and_filters_allowed_tools(
    tmp_path: Path,
) -> None:
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    step = ChainStepSpec(
        id="s1",
        kind="text",
        prompt_section="step:one",
        tool_choice=ToolChoice.model_validate(
            {
                "mode": "required",
                "allowed_tools": [{"name": "filesystem_list_files"}],
            }
        ),
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(),
        chain=[step],
        tools=["filesystem_list_files", "filesystem_read_text"],
        output=OutputSpec(file=None),
        tool_choice=ToolChoice.model_validate("none"),
    )

    agent = _make_agent(
        spec=spec,
        step_prompts={"step:one": "say hi"},
        tools_root=tools_root,
        tmp_path=tmp_path,
    )
    request = agent.batch()

    assert request.tool_choice is not None
    assert request.tool_choice.mode == "required"
    assert request.tools is not None
    assert len(request.tools) == 1
    assert request.tools[0].name == "filesystem_list_files"


def test_tool_choice_single_shot_uses_agent_level_none(tmp_path: Path) -> None:
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    spec = AgentSpec(
        name="test",
        model=ModelSpec(),
        chain=[],
        tools=["filesystem_list_files"],
        output=OutputSpec(file=None),
        tool_choice=ToolChoice.model_validate("none"),
    )

    agent = _make_agent(spec=spec, tools_root=tools_root, tmp_path=tmp_path)
    request = agent.batch()

    assert request.tool_choice is not None
    assert request.tool_choice.mode == "none"
    assert request.tools is None
    assert request.tool_use_enabled is False


def test_tool_choice_any_maps_to_auto_for_openai() -> None:
    request = BatchSubmitRequest(
        request_id="r1",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="https://api.openai.com/v1",
            model_name="gpt-4.1-mini",
        ),
        messages=[BatchMessage(role="user", content="hello")],
        tool_choice=ToolChoice(mode="any"),
    )
    assert request.openai_tool_choice() is None


def test_tool_choice_bedrock_converse_any() -> None:
    request = BatchSubmitRequest(
        request_id="r-tools",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="bedrock",
            base_url="http://x",
            model_name="amazon.nova-pro-v1:0",
            bedrock_request_mode="converse",
        ),
        messages=[BatchMessage(role="user", content="hello")],
        tools=[
            BatchToolDefinition(
                name="filesystem_list_files",
                description="List files",
                input_schema={
                    "type": "object",
                    "properties": {"directory": {"type": "string"}},
                    "required": ["directory"],
                },
            )
        ],
        tool_choice=ToolChoice(mode="any"),
        tool_use_enabled=True,
    )
    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    tool_config = model_input.get("toolConfig")
    assert isinstance(tool_config, dict)
    assert tool_config.get("toolChoice") == {"any": {}}


def test_tool_choice_bedrock_open_weight_any() -> None:
    request = BatchSubmitRequest(
        request_id="r-tools",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="bedrock",
            base_url="http://x",
            model_name="qwen2.5-72b-instruct",
            bedrock_request_mode="open_weight_invoke",
        ),
        messages=[BatchMessage(role="user", content="hello")],
        tool_choice=ToolChoice(mode="any"),
    )
    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    assert model_input.get("tool_choice") == "required"
