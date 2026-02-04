"""Pydantic model for output configuration."""

from __future__ import annotations

from pydantic import BaseModel


class OutputSpec(BaseModel):
    format: str = "json"  # "json" or "markdown"
    file: str | None = None
