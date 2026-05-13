from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from quick_agent.models.agent_spec import AgentSpec
from quick_agent.models.handoff_spec import HandoffSpec
from quick_agent.models.loaded_agent_file import LoadedAgentFile
from quick_agent.models.model_spec import ModelSpec
from quick_agent.models.output_spec import OutputSpec
from quick_agent.quick_agent import QuickAgent


@dataclass
class DummyLoaded:
    spec: AgentSpec


class ExampleSchema(BaseModel):
    x: int


def _make_quick_agent(*, output_format: str = "json", output_schema: str | None = None, schemas: dict[str, str] | None = None) -> QuickAgent:
    spec = AgentSpec(
        name="test",
        model=ModelSpec(base_url="https://api.openai.com/v1", model_name="m"),
        chain=[],
        schemas=schemas or {},
        output=OutputSpec(format=output_format, output_schema=output_schema),
        handoff=HandoffSpec(),
    )
    loaded = LoadedAgentFile.from_parts(
        spec=spec,
        instructions="",
        system_prompt="",
        step_prompts={},
    )
    agent = object.__new__(QuickAgent)
    agent.loaded = loaded
    return agent


def test_finalize_output_contract_json_dict_returns_same() -> None:
    qa = _make_quick_agent(output_format="json")
    result = qa._finalize_output_contract({"k": "v"})
    assert result == {"k": "v"}


def test_finalize_output_contract_valid_json_string_parses() -> None:
    qa = _make_quick_agent(output_format="json")
    result = qa._finalize_output_contract('{"k": "v"}')
    assert result == {"k": "v"}


def test_finalize_output_contract_invalid_json_string_uses_repair() -> None:
    qa = _make_quick_agent(output_format="json")
    result = qa._finalize_output_contract('{"k": "v",}')
    assert result == {"k": "v"}


def test_finalize_output_contract_non_object_string_raises() -> None:
    qa = _make_quick_agent(output_format="json")
    with pytest.raises(ValueError, match="JSON output must be a JSON object."):
        qa._finalize_output_contract('[1,2,3]')


def test_finalize_output_contract_markdown_requires_string() -> None:
    qa = _make_quick_agent(output_format="markdown")
    with pytest.raises(ValueError, match=r"Text output must be a string \(format=markdown"):
        qa._finalize_output_contract({"not": "string"})


def test_finalize_output_contract_text_requires_string() -> None:
    qa = _make_quick_agent(output_format="text")
    with pytest.raises(ValueError, match=r"Text output must be a string \(format=text"):
        qa._finalize_output_contract({"not": "string"})


def test_finalize_output_contract_structured_base_model_returns_same() -> None:
    schemas = {"ExampleSchema": "tests.test_quick_agent_output_contract:ExampleSchema"}
    qa = _make_quick_agent(output_format="structured", output_schema="ExampleSchema", schemas=schemas)
    output = ExampleSchema(x=1)
    result = qa._finalize_output_contract(output)
    assert isinstance(result, ExampleSchema)
    assert result.x == 1


def test_finalize_output_contract_structured_string_parses_against_schema() -> None:
    schemas = {"ExampleSchema": "tests.test_quick_agent_output_contract:ExampleSchema"}
    qa = _make_quick_agent(output_format="structured", output_schema="ExampleSchema", schemas=schemas)
    result = qa._finalize_output_contract('{"x": 2}')
    assert isinstance(result, ExampleSchema)
    assert result.x == 2
