"""Pydantic models for content processing config."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SampleSpec(BaseModel):
    ratios: tuple[int, int, int] = Field(default=(25, 50, 25))
    max_chunk_tokens: int
    debug_output_file: str | None = None


class ContentProcessingSpec(BaseModel):
    sample: SampleSpec | None = None
