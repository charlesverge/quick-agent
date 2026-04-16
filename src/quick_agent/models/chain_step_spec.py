"""Pydantic model for a single chain step."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ToolChoiceToolRef(BaseModel):
    """Reference to a tool in allowed_tools list."""

    type: Literal["function"] = "function"
    name: str


class ToolChoice(BaseModel):
    """Tool choice configuration for chain steps and agents."""

    mode: Literal["auto", "any", "required", "none"] | None = None
    type: Literal["function"] | None = None
    name: str | None = None
    allowed_tools: list[ToolChoiceToolRef] | None = None


class ChainStepSpec(BaseModel):
    id: str
    kind: str  # "text" or "structured" (you may extend: "parallel_map", "fanout", etc.)
    prompt_section: str
    output_schema: Optional[str] = None
    tool_choice: ToolChoice | None = None
