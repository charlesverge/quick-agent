"""Pydantic model for a single chain step."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="before")
    @classmethod
    def parse_shorthand(cls, value: object) -> object:
        if isinstance(value, str):
            return {"mode": value}
        return value

    @model_validator(mode="after")
    def validate_shape(self) -> "ToolChoice":
        if self.type == "function" and self.name is None:
            raise ValueError("tool_choice type='function' requires name.")
        if self.name is not None and self.type != "function":
            raise ValueError("tool_choice name requires type='function'.")
        if self.allowed_tools is not None and len(self.allowed_tools) == 0:
            raise ValueError("tool_choice allowed_tools cannot be empty.")
        return self


class ChainStepSpec(BaseModel):
    id: str
    kind: str  # "text" or "structured" (you may extend: "parallel_map", "fanout", etc.)
    prompt_section: str
    output_schema: Optional[str] = None
    tool_choice: ToolChoice | None = None
    max_tool_calls: int | None = Field(default=None, ge=1)
