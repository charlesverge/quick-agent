"""Pydantic model for a single chain step."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class ChainStepSpec(BaseModel):
    id: str
    kind: str  # "text" or "structured" (you may extend: "parallel_map", "fanout", etc.)
    prompt_section: str
    output_schema: Optional[str] = None
