"""Pydantic model for agent frontmatter spec."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from quick_agent.models.chain_step_spec import ChainStepSpec
from quick_agent.models.content_processing_spec import ContentProcessingSpec
from quick_agent.models.handoff_spec import HandoffSpec
from quick_agent.models.model_spec import ModelSpec
from quick_agent.models.output_spec import OutputSpec


class AgentSpec(BaseModel):
    name: str
    description: str = ""
    model: ModelSpec = Field(default_factory=ModelSpec)
    tools: list[str] = Field(default_factory=list)
    schemas: dict[str, str] = Field(default_factory=dict)  # alias -> "module:ClassName"
    chain: list[ChainStepSpec] = Field(default_factory=list)
    single_shot_use_pydantic_ai: bool = False
    output: OutputSpec = Field(default_factory=OutputSpec)
    handoff: HandoffSpec = Field(default_factory=HandoffSpec)
    content_processing: ContentProcessingSpec | None = None
    nested_output: Literal["inline", "file"] = "inline"
    safe_dir: str | None = None

    @model_validator(mode="after")
    def validate_output_schema_usage(self) -> "AgentSpec":
        if self.output.output_schema and self.chain:
            raise ValueError(
                "output.output_schema requires an empty chain (single-shot mode)."
            )
        return self
