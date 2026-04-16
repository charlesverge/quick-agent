from __future__ import annotations

from unittest.mock import MagicMock

from pydantic import BaseModel

from quick_agent.agent_execution_context import AgentExecutionContext
from quick_agent.models.model_spec import ModelSettings, ModelSpec


class SchemaA(BaseModel):
    name: str


class SchemaB(BaseModel):
    value: int


def _build_context(
    model_settings_json: ModelSettings,
) -> AgentExecutionContext:
    config = MagicMock()
    config.model_spec = ModelSpec(base_url="https://api.openai.com/v1")
    return AgentExecutionContext(
        config=config,
        extra_headers=None,
        extra_body=None,
        model_settings_json=model_settings_json,
        http_client=None,
        model_name="gpt-4",
        client=None,
        effective_base_url="https://api.openai.com/v1",
    )


class TestBuildStructuredModelSettings:
    def test_sets_response_format_for_schema(self) -> None:
        ctx = _build_context(model_settings_json=ModelSettings())
        result = ctx.build_structured_model_settings(schema_cls=SchemaA)
        assert result is not None
        extra_body = result.extra_body
        assert isinstance(extra_body, dict)
        rf = extra_body["response_format"]
        assert isinstance(rf, dict)
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "SchemaA"

    def test_second_schema_overwrites_response_format(self) -> None:
        ctx = _build_context(
            model_settings_json=ModelSettings(extra_body={"stream": False})
        )
        ctx.build_structured_model_settings(schema_cls=SchemaA)
        result = ctx.build_structured_model_settings(schema_cls=SchemaB)
        assert result is not None
        extra_body = result.extra_body
        assert isinstance(extra_body, dict)
        rf = extra_body["response_format"]
        assert isinstance(rf, dict)
        assert rf["json_schema"]["name"] == "SchemaB"

    def test_cached_settings_not_mutated(self) -> None:
        original = ModelSettings(extra_body={"stream": False})
        ctx = _build_context(model_settings_json=original)
        ctx.build_structured_model_settings(schema_cls=SchemaA)
        extra_body = original.extra_body
        assert isinstance(extra_body, dict)
        assert "response_format" not in extra_body
