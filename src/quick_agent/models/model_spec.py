"""Pydantic model for LLM configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue


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
