"""Pydantic model for LLM configuration."""

from __future__ import annotations

from typing import Optional

import httpx
import openai
from openai._types import Headers
from pydantic import BaseModel, Field, JsonValue
from quick_agent.models.chain_step_spec import ToolChoice


class ModelSettings(BaseModel):
    """Settings to configure an LLM."""

    max_completion_tokens: Optional[int] | openai.Omit = openai.omit
    temperature: Optional[float] | openai.Omit = openai.omit
    top_p: Optional[float] | openai.Omit = openai.omit
    timeout: float | httpx.Timeout | None = None
    parallel_tool_calls: bool | openai.Omit = openai.omit
    seed: Optional[int] | openai.Omit = openai.omit
    presence_penalty: Optional[float] | openai.Omit = openai.omit
    frequency_penalty: Optional[float] | openai.Omit = openai.omit
    logit_bias: Optional[dict[str, int]] | openai.Omit = openai.omit
    stop: Optional[list[str]] | openai.Omit = openai.omit
    extra_headers: Optional[Headers] | None = None
    thinking: bool | str | None = None
    tool_choice: ToolChoice | None = None
    response_as_tool: bool | None = None
    extra_body: Optional[object] | openai.Omit = openai.omit

    model_config = {"extra": "allow", "arbitrary_types_allowed": True}

    def __eq__(self, other: object) -> bool:
        if isinstance(other, dict):
            raise NotImplementedError(
                "Comparison to dict is not supported for ModelSettings"
            )
        if isinstance(other, ModelSettings):
            return self.model_dump(exclude_defaults=True) == other.model_dump(
                exclude_defaults=True
            )
        return False


class ModelSpec(BaseModel):
    provider: str = Field(default="openai-compatible")
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key_env: str = Field(default="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-5.2")
    temperature: float = 0.2
    max_completion_tokens: int = 2048
    num_ctx: int | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)
    keepalive_expiry_seconds: float | None = Field(default=None, gt=0)
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, JsonValue] | None = None
    convert_null: bool | None = Field(default=None)
