"""Pydantic model for output configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class OutputSpec(BaseModel):
    format: str = "json"  # "json" or "markdown" or "structured"
    file: str | None = None
    output_schema: str | None = None
    return_compiled_output: bool = False
    compiled_schema: str | None = Field(default=None, alias="schema")
