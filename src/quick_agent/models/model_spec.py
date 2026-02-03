"""Pydantic model for LLM configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelSpec(BaseModel):
    provider: str = Field(default="openai-compatible")
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key_env: str = Field(default="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-5.2")
    temperature: float = 0.2
    max_tokens: int = 2048
