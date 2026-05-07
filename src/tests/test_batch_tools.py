import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, JsonValue

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.executor import ToolCallResult
from quick_agent.input_adaptors import TextInput
from quick_agent.models import (
    AgentSpec,
    ChainStepSpec,
    ChunkProcessingSpec,
    ContentProcessingSpec,
    LoadedAgentFile,
    ModelSpec,
    SampleSpec,
)
from quick_agent.models.batch_request import (
    BatchImportRequest,
    BatchMessage,
    BatchModelConfig,
    BatchSubmitRequest,
    BatchToolDefinition,
)
from quick_agent.models.chain_step_spec import ToolChoice
from quick_agent.models.output_spec import OutputSpec
from quick_agent.models.run_input import RunInput
from quick_agent.quick_agent import QuickAgent, QuickAgentConfigError

_CLASSIFICATION_RESULT_RESPONSE_FORMAT: dict[str, JsonValue] = {
    "type": "json_schema",
    "json_schema": {
        "name": "ClassificationResult",
        "schema": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["label", "confidence"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class FakeRegistry(AgentRegistry):
    def __init__(self, loaded: LoadedAgentFile) -> None:
        super().__init__([])
        self._loaded = loaded

    def get(self, agent_id: str) -> LoadedAgentFile:
        return self._loaded


def _make_loaded_with_chain(
    chain: list[ChainStepSpec],
    *,
    schemas: dict[str, str] | None = None,
    output: OutputSpec | None = None,
) -> LoadedAgentFile:
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=chain,
        schemas=schemas or {},
        output=output or OutputSpec(file="out/result.json"),
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
    memory: dict[str, Any] | None = None,
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
        input_data=TextInput(run_input.text),
        extra_tools=None,
        model=loaded.spec.model,
        write_output=False,
        record_http_traffic=False,
        memory=memory,
        test_mode=True,
    )
    agent.run_input = run_input
    agent.state = {"agent_id": "a", "steps": {}, "last_step_output": None}
    return agent


class ExampleSchema(BaseModel):
    x: int


def test_create_batch_request_for_current_step() -> None:
    qa = _make_quick_agent_for_test()
    request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="do thing",
        system_prompt="system prompt",
        user_prompt="input prompt",
        model_settings=qa._executor.context.model_settings_json,
    )
    assert request.agent_id == "a"
    assert request.step_id == "s1"
    assert request.step_kind == "text"
    assert request.model.model_name == qa.model_spec.model_name
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert request.messages[1].content == "input prompt"


def test_create_batch_request_includes_tool_definitions() -> None:
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test()
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]

    request = qa.create_batch_request_for_current_step(
        step_id=None,
        step_kind="single_shot",
        output_schema=None,
        instructions=None,
        system_prompt="sys",
        user_prompt="input",
        model_settings=qa._executor.context.model_settings_json,
    )

    assert request.tool_use_enabled is True
    assert request.tools is not None
    assert len(request.tools) == 1
    assert request.tools[0].name == "filesystem_list_files"
    properties = request.tools[0].input_schema.get("properties")
    assert isinstance(properties, dict)
    assert "directory" in properties


def test_create_batch_request_bedrock_tool_schema_is_strict() -> None:
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test()
    qa.model_spec = ModelSpec(provider="bedrock", base_url="http://x", model_name="m")
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]

    request = qa.create_batch_request_for_current_step(
        step_id=None,
        step_kind="single_shot",
        output_schema=None,
        instructions=None,
        system_prompt="sys",
        user_prompt="input",
        model_settings=qa._executor.context.model_settings_json,
    )

    assert request.tools is not None
    assert request.tools[0].strict is True
    assert request.tools[0].input_schema["additionalProperties"] is False


def test_create_batch_request_uses_model_settings_tool_choice_any() -> None:
    qa = _make_quick_agent_for_test()
    qa.loaded.spec.tool_choice = ToolChoice(mode="any")
    request = qa.batch()[0]
    assert request.tool_choice is not None
    assert request.tool_choice.mode == "any"


def test_create_batch_request_uses_model_settings_tool_choice_allowed_tools() -> None:
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test()
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files", "filesystem_read_text"]

    request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="do thing",
        system_prompt="system prompt",
        user_prompt="input prompt",
        model_settings=qa._executor.context.model_settings_json.model_copy(
            update={
                "tool_choice": ToolChoice.model_validate(
                    {
                        "mode": "required",
                        "allowed_tools": [{"name": "filesystem_list_files"}],
                    }
                )
            }
        ),
    )

    assert request.tool_choice is not None
    assert request.tool_choice.mode == "required"
    assert request.tools is not None
    assert len(request.tools) == 1
    assert request.tools[0].name == "filesystem_list_files"


def test_batch_submit_converse_jsonl_includes_tool_config() -> None:
    request = BatchSubmitRequest(
        request_id="r-tools",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="http://x",
            model_name="m",
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
        tool_use_enabled=True,
    )

    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    tool_config = model_input.get("toolConfig")
    assert isinstance(tool_config, dict)
    tools = tool_config.get("tools")
    assert isinstance(tools, list)
    assert len(tools) == 1
    assert isinstance(tools[0], dict)
    assert tools[0]["toolSpec"]["name"] == "filesystem_list_files"


def test_batch_submit_converse_jsonl_includes_tool_strict_flag() -> None:
    request = BatchSubmitRequest(
        request_id="r-tools-strict",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="bedrock",
            base_url="http://x",
            model_name="m",
            bedrock_request_mode="converse",
        ),
        messages=[BatchMessage(role="user", content="hello")],
        tools=[
            BatchToolDefinition(
                name="filesystem_list_files",
                description="List files",
                input_schema={"type": "object", "properties": {}},
                strict=True,
            )
        ],
        tool_use_enabled=True,
    )

    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    tool_config = model_input.get("toolConfig")
    assert isinstance(tool_config, dict)
    tools_obj = tool_config.get("tools")
    assert isinstance(tools_obj, list)
    assert isinstance(tools_obj[0], dict)
    assert tools_obj[0]["toolSpec"]["strict"] is True


def test_batch_submit_jsonl_line_rejects_non_strict_bedrock_tool() -> None:
    request = BatchSubmitRequest(
        request_id="r-tools-not-strict",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="bedrock",
            base_url="http://x",
            model_name="m",
            bedrock_request_mode="converse",
        ),
        messages=[BatchMessage(role="user", content="hello")],
        tools=[
            BatchToolDefinition(
                name="filesystem_list_files",
                description="List files",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                strict=False,
            )
        ],
        tool_use_enabled=True,
    )

    with pytest.raises(ValueError, match="must set strict=true"):
        _ = request.jsonl_line


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
            bedrock_request_mode="open_weight_invoke",
            extra_body={
                "inferenceConfig": {"topP": 0.9, "max_new_tokens": 300},
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
    assert model_input["system"] == [{"text": "sys"}]
    messages = model_input["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {
        "role": "user",
        "content": [{"text": "hello"}],
    }


def test_batch_submit_jsonl_line_omits_converse_system_and_output_for_noop() -> None:
    request = BatchSubmitRequest(
        request_id="bedrock-20260423T134748Z-1776952068677:noop:9",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="http://x",
            model_name="m",
            bedrock_request_mode="converse",
        ),
        response_format={"type": "json_schema"},
        messages=[
            BatchMessage(role="system", content="sys"),
            BatchMessage(role="user", content="say ok"),
        ],
    )
    model_input = request.jsonl_line["modelInput"]
    assert isinstance(model_input, dict)
    assert "system" not in model_input
    assert "outputConfig" not in model_input


def test_batch_submit_jsonl_line_converts_converse_response_format() -> None:
    request = BatchSubmitRequest(
        request_id="r-converse-structured",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="http://x",
            model_name="m",
            bedrock_request_mode="converse",
        ),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "CompanyPageRouterOutput",
                "schema": {
                    "type": "object",
                    "properties": {"page_type": {"type": "string"}},
                    "required": ["page_type"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        messages=[BatchMessage(role="user", content="hello")],
    )
    model_input = request.jsonl_line["modelInput"]
    assert isinstance(model_input, dict)
    output_config = model_input["outputConfig"]
    assert isinstance(output_config, dict)
    text_format = output_config["textFormat"]
    assert isinstance(text_format, dict)
    assert text_format["type"] == "json_schema"
    structure = text_format["structure"]
    assert isinstance(structure, dict)
    json_schema_obj = structure["jsonSchema"]
    assert isinstance(json_schema_obj, dict)
    assert json_schema_obj["name"] == "CompanyPageRouterOutput"
    assert json_schema_obj["schema"] == json.dumps(
        {
            "type": "object",
            "properties": {"page_type": {"type": "string"}},
            "required": ["page_type"],
            "additionalProperties": False,
        }
    )


def test_batch_submit_jsonl_line_rejects_unsupported_bedrock_schema_keyword() -> None:
    request = BatchSubmitRequest(
        request_id="r-converse-invalid-schema",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="bedrock",
            base_url="http://x",
            model_name="m",
            bedrock_request_mode="converse",
        ),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "InvalidSchema",
                "schema": {
                    "type": "object",
                    "properties": {"page_type": {"type": "string", "minLength": 1}},
                    "required": ["page_type"],
                    "additionalProperties": False,
                },
            },
        },
        messages=[BatchMessage(role="user", content="hello")],
    )

    with pytest.raises(ValueError, match="minLength: minLength is not supported"):
        _ = request.jsonl_line


def test_batch_submit_jsonl_line_rejects_external_ref_for_bedrock() -> None:
    request = BatchSubmitRequest(
        request_id="r-converse-external-ref",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="bedrock",
            base_url="http://x",
            model_name="m",
            bedrock_request_mode="converse",
        ),
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "InvalidSchema",
                "schema": {"$ref": "https://example.com/schema.json"},
            },
        },
        messages=[BatchMessage(role="user", content="hello")],
    )

    with pytest.raises(ValueError, match="external \\$ref values are not supported"):
        _ = request.jsonl_line


def test_batch_submit_jsonl_line_rejects_bare_json_schema_type() -> None:
    request = BatchSubmitRequest(
        request_id="r-bare-schema",
        agent_id="a",
        step_id=None,
        step_kind="single_shot",
        model=BatchModelConfig(
            provider="bedrock",
            base_url="http://x",
            model_name="m",
            bedrock_request_mode="converse",
        ),
        response_format={"type": "json_schema"},
        messages=[BatchMessage(role="user", content="hello")],
    )
    with pytest.raises(ValueError, match="json_schema or response_format.structure.jsonSchema"):
        _ = request.jsonl_line


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
        response_format=_CLASSIFICATION_RESULT_RESPONSE_FORMAT,
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
    assert output_config["format"] == _CLASSIFICATION_RESULT_RESPONSE_FORMAT


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
        response_format=_CLASSIFICATION_RESULT_RESPONSE_FORMAT,
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
        response_format=_CLASSIFICATION_RESULT_RESPONSE_FORMAT,
        messages=[BatchMessage(role="user", content="hello")],
    )
    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    assert "response_format" in model_input
    assert "output_config" not in model_input


def test_batch_entry_point_returns_submit_request() -> None:
    qa = _make_quick_agent_for_test()
    request = qa.batch()[0]
    assert request.step_id == "s1"
    assert request.step_kind == "text"


def test_batch_applies_sample_to_request_text() -> None:
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

    request = qa.batch()[0]

    assert request.context.input_text == "one two three"
    assert request.messages[1].content is not None
    assert "one two three" in request.messages[1].content
    assert "four five six" not in request.messages[1].content


def test_batch_chunk_processing_returns_request_items() -> None:
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
            )
        ),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="Summarize.",
        system_prompt="",
        step_prompts={},
    )
    run_input = RunInput(
        source_path="in.txt",
        kind="text",
        text="one two three four five six seven eight nine",
        data=None,
    )
    qa = _make_quick_agent_for_test(loaded=loaded, run_input=run_input)

    requests = qa.batch()

    assert len(requests) > 1
    index = 0
    while index < len(requests):
        request = requests[index]
        assert request.context.execution_mode == "chunk"
        assert request.context.item_index == index
        assert request.context.item_count == len(requests)
        assert request.context.input_text
        index += 1


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
    request = qa.batch()[0]
    assert request.response_format is not None
    assert request.response_format["type"] == "json_schema"


def test_create_batch_request_openai_structured_tools_response_as_tool_true_uses_final_result() -> (
    None
):
    step = ChainStepSpec(
        id="s1",
        kind="structured",
        prompt_section="step:one",
        output_schema="Out",
        response_as_tool=True,
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(
            provider="openai-compatible", base_url="http://x", model_name="m"
        ),
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
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]

    request = qa.batch()[0]

    assert request.response_format is None
    assert request.final_result_tool_enabled is True
    assert request.tools is not None
    names = [tool.name for tool in request.tools]
    assert "filesystem_list_files" in names
    assert "final_result" in names
    strict_flags = [tool.strict for tool in request.tools]
    assert all(strict_flags)


def test_create_batch_request_bedrock_structured_tools_defaults_response_as_tool_true() -> (
    None
):
    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Out"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(provider="bedrock", base_url="http://x", model_name="m"),
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
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]

    request = qa.batch()[0]

    assert request.response_format is None
    assert request.response_as_tool is True
    assert request.final_result_tool_enabled is True
    assert request.tools is not None
    names = [tool.name for tool in request.tools]
    assert "filesystem_list_files" in names
    assert "final_result" in names
    strict_flags = [tool.strict for tool in request.tools]
    assert all(strict_flags)


def test_create_batch_request_bedrock_structured_tools_response_as_tool_false_raises() -> (
    None
):
    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Out"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(provider="bedrock", base_url="http://x", model_name="m"),
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
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]

    with pytest.raises(ValueError, match="response_as_tool=true"):
        qa.create_batch_request_for_current_step(
            step_id="s1",
            step_kind="structured",
            output_schema="Out",
            instructions="system",
            system_prompt="",
            user_prompt="input",
            model_settings=qa._executor.context.build_structured_model_settings(
                schema_cls=ExampleSchema
            ).model_copy(update={"response_as_tool": False}),
        )


def test_create_batch_request_open_weight_structured_tools_response_as_tool_false_raises() -> (
    None
):
    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Out"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(
            provider="bedrock",
            base_url="http://x",
            model_name="qwen.qwen3-next-80b-a3b",
        ),
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
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]
    with pytest.raises(ValueError, match="response_as_tool=true"):
        qa.create_batch_request_for_current_step(
            step_id="s1",
            step_kind="structured",
            output_schema="Out",
            instructions="system",
            system_prompt="",
            user_prompt="input",
            model_settings=qa._executor.context.build_structured_model_settings(
                schema_cls=ExampleSchema
            ).model_copy(update={"response_as_tool": False}),
        )


def test_create_batch_request_open_weight_structured_tools_response_as_tool_true_omits_response_format() -> (
    None
):
    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Out"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(
            provider="bedrock",
            base_url="http://x",
            model_name="qwen.qwen3-next-80b-a3b",
        ),
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
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]
    request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="structured",
        output_schema="Out",
        instructions="system",
        system_prompt="",
        user_prompt="input",
        model_settings=qa._executor.context.build_structured_model_settings(
            schema_cls=ExampleSchema
        ).model_copy(update={"response_as_tool": True, "bedrock_request_mode": "open_weight_invoke"}),
    )
    assert request.response_format is None
    assert request.final_result_tool_enabled is True
    line = request.jsonl_line
    model_input = line["modelInput"]
    assert isinstance(model_input, dict)
    assert "response_format" not in model_input
    tools_obj = model_input["tools"]
    assert isinstance(tools_obj, list)
    names = [tool["function"]["name"] for tool in tools_obj if isinstance(tool, dict)]
    assert "final_result" in names


def test_create_batch_request_non_bedrock_structured_tools_response_as_tool_true_uses_final_result() -> (
    None
):
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
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]

    request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="structured",
        output_schema="Out",
        instructions="system",
        system_prompt="",
        user_prompt="input",
        model_settings=qa._executor.context.build_structured_model_settings(
            schema_cls=ExampleSchema
        ).model_copy(update={"response_as_tool": True}),
    )

    assert request.response_format is None
    assert request.final_result_tool_enabled is True
    assert request.tools is not None
    names = [tool.name for tool in request.tools]
    assert "final_result" in names


def test_batch_request_raises_when_agent_response_as_tool_false_with_tools() -> None:
    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Out"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        schemas={"Out": f"{__name__}:ExampleSchema"},
        output=OutputSpec(file=None),
        response_as_tool=False,
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]

    with pytest.raises(QuickAgentConfigError, match="response_as_tool=true"):
        qa.batch()


def test_batch_uses_chain_response_as_tool_override() -> None:
    step = ChainStepSpec(
        id="s1",
        kind="structured",
        prompt_section="step:one",
        output_schema="Out",
        response_as_tool=True,
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        schemas={"Out": f"{__name__}:ExampleSchema"},
        output=OutputSpec(file=None),
        response_as_tool=False,
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]

    request = qa.batch()[0]
    assert request.response_as_tool is True
    assert request.final_result_tool_enabled is True


@pytest.mark.parametrize("response_as_tool", [True, False])
def test_create_batch_request_bedrock_structured_no_tools_preserves_response_format(
    response_as_tool: bool,
) -> None:
    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Out"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(provider="bedrock", base_url="http://x", model_name="m"),
        chain=[step],
        schemas={"Out": f"{__name__}:ExampleSchema"},
        output=OutputSpec(file=None),
        response_as_tool=response_as_tool,
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    qa = _make_quick_agent_for_test(loaded=loaded)

    request = qa.batch()[0]
    assert request.response_format is not None
    assert request.final_result_tool_enabled is False
    assert request.tools is None


@pytest.mark.parametrize("response_as_tool", [True, False])
def test_create_batch_request_bedrock_no_structured_with_tools_preserves_tool_behavior(
    response_as_tool: bool,
) -> None:
    spec = AgentSpec(
        name="test",
        model=ModelSpec(provider="bedrock", base_url="http://x", model_name="m"),
        chain=[ChainStepSpec(id="s1", kind="text", prompt_section="step:one")],
        response_as_tool=response_as_tool,
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]

    request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="system",
        system_prompt="",
        user_prompt="input",
        model_settings=qa._executor.context.model_settings_json.model_copy(
            update={"response_as_tool": response_as_tool}
        ),
    )
    assert request.response_format is None
    assert request.final_result_tool_enabled is False
    assert request.tool_use_enabled is True
    assert request.tools is not None
    names = [tool.name for tool in request.tools]
    assert "final_result" not in names


@pytest.mark.parametrize("response_as_tool", [True, False])
def test_create_batch_request_non_bedrock_structured_no_tools_preserves_response_format(
    response_as_tool: bool,
) -> None:
    step = ChainStepSpec(
        id="s1", kind="structured", prompt_section="step:one", output_schema="Out"
    )
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[step],
        schemas={"Out": f"{__name__}:ExampleSchema"},
        output=OutputSpec(file=None),
        response_as_tool=response_as_tool,
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    qa = _make_quick_agent_for_test(loaded=loaded)

    request = qa.batch()[0]
    assert request.response_format is not None
    assert request.final_result_tool_enabled is False
    assert request.tools is None


@pytest.mark.parametrize("response_as_tool", [True, False])
def test_create_batch_request_non_bedrock_no_structured_with_tools_preserves_tool_behavior(
    response_as_tool: bool,
) -> None:
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="http://x", model_name="m"),
        chain=[ChainStepSpec(id="s1", kind="text", prompt_section="step:one")],
        response_as_tool=response_as_tool,
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="system",
        system_prompt="",
        step_prompts={"step:one": "do thing"},
    )
    tools_root = Path(__file__).parent.parent / "quick_agent" / "tools"
    qa = _make_quick_agent_for_test(loaded=loaded)
    qa._tools = AgentTools([tools_root])
    qa.tool_ids = ["filesystem_list_files"]

    request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="system",
        system_prompt="",
        user_prompt="input",
        model_settings=qa._executor.context.model_settings_json.model_copy(
            update={"response_as_tool": response_as_tool}
        ),
    )
    assert request.response_format is None
    assert request.final_result_tool_enabled is False
    assert request.tool_use_enabled is True
    assert request.tools is not None
    names = [tool.name for tool in request.tools]
    assert "final_result" not in names


def test_import_outcome_handles_tool_use_state() -> None:
    qa = _make_quick_agent_for_test()
    submit_request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="do thing",
        system_prompt="system",
        user_prompt="user prompt",
        model_settings=qa._executor.context.model_settings_json,
    )
    batch_import = BatchImportRequest(
        request_id="r1",
        payload={
            "state": "tool_use",
            "tool_calls": [
                {
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "tool_name", "arguments": {"x": 1}},
                }
            ],
            "submit_request": submit_request.model_dump(mode="json"),
        },
    )
    outcome = qa._executor.import_outcome(batch_import=batch_import)
    assert outcome.tool_calls is not None
    assert len(outcome.tool_calls) == 1
    assert outcome.tool_calls[0]["id"] == "call1"
    assert outcome.pending_submit_request is not None


def test_import_outcome_maps_final_result_tool_call_to_completed_result() -> None:
    qa = _make_quick_agent_for_test()
    submit_request = BatchSubmitRequest(
        request_id="r-final",
        agent_id="a",
        step_id="s1",
        step_kind="structured",
        output_schema="Out",
        model=BatchModelConfig(
            provider="bedrock",
            base_url="http://x",
            model_name="m",
            bedrock_request_mode="converse",
        ),
        messages=[BatchMessage(role="user", content="hello")],
        tools=[
            BatchToolDefinition(
                name="final_result",
                description="final",
                input_schema={
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                },
                strict=True,
            )
        ],
        tool_use_enabled=True,
        response_as_tool=True,
        final_result_tool_enabled=True,
    )
    batch_import = BatchImportRequest(
        request_id="r-final",
        payload={
            "state": "tool_use",
            "tool_calls": [
                {
                    "id": "call1",
                    "name": "final_result",
                    "arguments": {"x": 11},
                }
            ],
            "submit_request": submit_request.model_dump(mode="json"),
        },
    )
    outcome = qa._executor.import_outcome(batch_import=batch_import)
    assert outcome.result == {"x": 11}
    assert outcome.tool_calls is None


def test_import_outcome_non_matching_final_result_contract_keeps_tool_loop() -> None:
    qa = _make_quick_agent_for_test()
    submit_request = BatchSubmitRequest(
        request_id="r-tool-loop",
        agent_id="a",
        step_id="s1",
        step_kind="structured",
        output_schema="Out",
        model=BatchModelConfig(
            provider="bedrock",
            base_url="http://x",
            model_name="m",
            bedrock_request_mode="converse",
        ),
        messages=[BatchMessage(role="user", content="hello")],
        tools=[
            BatchToolDefinition(
                name="final_result",
                description="final",
                input_schema={
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                },
                strict=True,
            ),
            BatchToolDefinition(
                name="filesystem_list_files",
                description="List files",
                input_schema={
                    "type": "object",
                    "properties": {"directory": {"type": "string"}},
                },
                strict=True,
            ),
        ],
        tool_use_enabled=True,
        response_as_tool=True,
        final_result_tool_enabled=True,
    )
    batch_import = BatchImportRequest(
        request_id="r-tool-loop",
        payload={
            "state": "tool_use",
            "tool_calls": [
                {
                    "id": "call1",
                    "name": "filesystem_list_files",
                    "arguments": {"directory": "."},
                }
            ],
            "submit_request": submit_request.model_dump(mode="json"),
        },
    )
    outcome = qa._executor.import_outcome(batch_import=batch_import)
    assert outcome.result is None
    assert outcome.tool_calls is not None
    assert outcome.pending_submit_request is not None


def test_import_outcome_tool_use_missing_tool_calls_raises() -> None:
    qa = _make_quick_agent_for_test()
    batch_import = BatchImportRequest(
        request_id="r1",
        payload={"state": "tool_use"},
    )
    with pytest.raises(ValueError, match="tool_calls"):
        qa._executor.import_outcome(batch_import=batch_import)


@pytest.mark.anyio
async def test_import_result_tool_use_enforces_max_tool_calls() -> None:
    qa = _make_quick_agent_for_test()
    submit_request = qa.batch()[0].model_copy(update={"max_tool_calls": 1})
    submit_request.messages = [
        BatchMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "call1",
                    "type": "function",
                    "function": {"name": "random_get_name", "arguments": "{}"},
                }
            ],
        )
    ]
    with pytest.raises(ValueError, match="Max tool call rounds reached"):
        await qa.import_result(
            batch_import=BatchImportRequest(
                request_id=submit_request.request_id,
                payload={
                    "state": "tool_use",
                    "tool_calls": [{"id": "call2", "name": "random_get_name"}],
                    "submit_request": submit_request.model_dump(mode="json"),
                },
            )
        )


def test_import_outcome_handles_openai_gpt_5_2_tool_call_format() -> None:
    qa = _make_quick_agent_for_test()
    submit_request = qa.create_batch_request_for_current_step(
        step_id="summary_generation",
        step_kind="text",
        output_schema=None,
        instructions="summarize",
        system_prompt="system",
        user_prompt="user prompt",
        model_settings=qa._executor.context.model_settings_json,
    )
    batch_import = BatchImportRequest(
        request_id="r1",
        payload={
            "state": "tool_use",
            "tool_calls": [
                {
                    "id": "call_WteJFS1sxnJQaP4dlwPhC1Ka",
                    "name": "retrieve_markdown_summaries_for_urls",
                    "arguments": json.dumps(
                        {
                            "urls": [
                                "https://acme.example.com/",
                                "https://acme.example.com/about",
                                "https://acme.example.com/careers",
                            ]
                        }
                    ),
                }
            ],
            "submit_request": submit_request.model_dump(mode="json"),
        },
    )

    outcome = qa._executor.import_outcome(batch_import=batch_import)

    assert outcome.tool_calls is not None
    assert len(outcome.tool_calls) == 1
    assert outcome.tool_calls[0]["id"] == "call_WteJFS1sxnJQaP4dlwPhC1Ka"
    assert outcome.tool_calls[0]["name"] == "retrieve_markdown_summaries_for_urls"
    assert isinstance(outcome.tool_calls[0]["arguments"], dict)
    assert outcome.tool_calls[0]["arguments"] == {
        "urls": [
            "https://acme.example.com/",
            "https://acme.example.com/about",
            "https://acme.example.com/careers",
        ]
    }
    assert outcome.pending_submit_request is not None


def test_build_next_request_with_tool_results_extends_messages() -> None:
    qa = _make_quick_agent_for_test()
    submit_request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="do thing",
        system_prompt="system",
        user_prompt="user prompt",
        model_settings=qa._executor.context.model_settings_json,
    )
    tool_calls: list[dict[str, object]] = [
        {"id": "call1", "name": "tool_name", "arguments": {"x": 1}}
    ]
    executed = [ToolCallResult(id="call1", name="tool_name", content="result text")]
    next_req = qa._executor._build_next_request_with_tool_results(
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


def test_build_next_request_with_tool_results_clears_tool_choice() -> None:
    qa = _make_quick_agent_for_test()
    submit_request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="do thing",
        system_prompt="system",
        user_prompt="user prompt",
        model_settings=qa._executor.context.model_settings_json,
    )
    tool_calls: list[dict[str, object]] = [
        {"id": "call1", "name": "tool_name", "arguments": {"x": 1}}
    ]
    executed = [ToolCallResult(id="call1", name="tool_name", content="result text")]
    for mode in ("required", "any"):
        next_req = qa._executor._build_next_request_with_tool_results(
            tool_calls=tool_calls,
            executed=executed,
            submit_request=submit_request.model_copy(
                update={"tool_choice": ToolChoice(mode=mode)}
            ),
        )
        assert next_req.tool_choice is None


def test_build_next_request_with_tool_results_keeps_non_required_tool_choice() -> None:
    qa = _make_quick_agent_for_test()
    submit_request = qa.create_batch_request_for_current_step(
        step_id="s1",
        step_kind="text",
        output_schema=None,
        instructions="do thing",
        system_prompt="system",
        user_prompt="user prompt",
        model_settings=qa._executor.context.model_settings_json,
    )
    submit_request = submit_request.model_copy(
        update={"tool_choice": ToolChoice(mode="none"), "max_tool_calls": 5}
    )
    tool_calls: list[dict[str, object]] = [
        {"id": "call1", "name": "tool_name", "arguments": {"x": 1}}
    ]
    executed = [ToolCallResult(id="call1", name="tool_name", content="result text")]
    next_req = qa._executor._build_next_request_with_tool_results(
        tool_calls=tool_calls,
        executed=executed,
        submit_request=submit_request,
    )
    assert next_req.tool_choice is not None
    assert next_req.tool_choice.mode == "none"
    assert next_req.max_tool_calls == 5


def test_build_next_request_with_tool_results_keeps_response_format_none_for_final_result_mode() -> (
    None
):
    qa = _make_quick_agent_for_test()
    submit_request = BatchSubmitRequest(
        request_id="r1",
        agent_id="a",
        step_id="s1",
        step_kind="structured",
        output_schema="Out",
        model=BatchModelConfig(
            provider="openai-compatible",
            base_url="http://x",
            model_name="qwen.qwen3-next-80b-a3b",
            bedrock_request_mode="open_weight_invoke",
        ),
        messages=[BatchMessage(role="user", content="hello")],
        response_format=None,
        tool_choice=ToolChoice(mode="required"),
        tools=[
            BatchToolDefinition(
                name="final_result",
                description="Return structured output",
                input_schema={"type": "object"},
                strict=True,
            )
        ],
        tool_use_enabled=True,
        response_as_tool=True,
        final_result_tool_enabled=True,
    )
    tool_calls: list[dict[str, object]] = [
        {"id": "call1", "name": "final_result", "arguments": {"name": "Ethan"}}
    ]
    executed = [ToolCallResult(id="call1", name="final_result", content="done")]
    next_req = qa._executor._build_next_request_with_tool_results(
        tool_calls=tool_calls,
        executed=executed,
        submit_request=submit_request,
    )
    assert next_req.response_format is None
    assert next_req.tool_choice is None


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
