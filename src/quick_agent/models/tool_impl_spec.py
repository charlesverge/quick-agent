"""Pydantic model for tool implementation metadata."""

from __future__ import annotations

from pydantic import BaseModel


class ToolImplSpec(BaseModel):
    kind: str  # "python"
    module: str
    function: str
