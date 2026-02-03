"""Pydantic model for agent frontmatter spec."""

from __future__ import annotations

from pydantic import BaseModel, Field

from quick_agent.models.chain_step_spec import ChainStepSpec
from quick_agent.models.handoff_spec import HandoffSpec
from quick_agent.models.model_spec import ModelSpec
from quick_agent.models.output_spec import OutputSpec


class AgentSpec(BaseModel):
    name: str
    description: str = ""
    model: ModelSpec = Field(default_factory=ModelSpec)
    tools: list[str] = Field(default_factory=list)
    schemas: dict[str, str] = Field(default_factory=dict)  # alias -> "module:ClassName"
    chain: list[ChainStepSpec]
    output: OutputSpec = Field(default_factory=OutputSpec)
    handoff: HandoffSpec = Field(default_factory=HandoffSpec)
    safe_dir: str | None = None
