"""Pydantic model for tool.json files."""

from __future__ import annotations

from pydantic import BaseModel

from quick_agent.models.tool_impl_spec import ToolImplSpec


class ToolJson(BaseModel):
    id: str
    name: str
    description: str = ""
    impl: ToolImplSpec
