"""Pydantic models for content processing config."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SampleSpec(BaseModel):
    ratios: tuple[int, int, int] = Field(default=(25, 50, 25))
    max_chunk_tokens: int
    debug_output_file: str | None = None


class ChunkProcessingSpec(BaseModel):
    mode: Literal["map_chunks", "map_paragraphs"]
    provider: Literal["semchunks"]
    max_chunk_tokens: int
    overlap_percent: int = Field(default=0, ge=0, le=20)
    overlap_token: int | None = Field(default=None, ge=0)
    max_output_items: int | None = Field(default=30, ge=1)
    max_output_tokens: int | None = Field(default=None, ge=1)


class ContentProcessingSpec(BaseModel):
    sample: SampleSpec | None = None
    chunk_processing: ChunkProcessingSpec | None = None
