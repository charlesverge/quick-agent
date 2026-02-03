"""Pydantic model for runtime input."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel


class RunInput(BaseModel):
    source_path: str
    kind: str  # "json" or "text"
    text: str
    data: Optional[dict[str, Any]] = None
