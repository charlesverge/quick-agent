"""Pydantic model for tool.json files."""

from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue

from quick_agent.models.tool_impl_spec import ToolImplSpec


class ToolJson(BaseModel):
    name: str
    description: str = ""
    impl: ToolImplSpec
    input_schema: dict[str, JsonValue] = Field(default_factory=dict)
