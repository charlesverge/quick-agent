"""Pydantic model for handoff configuration."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class HandoffSpec(BaseModel):
    enabled: bool = False
    agent_id: Optional[str] = None
    input_mode: str = "final_output_json"
