"""Loaded agent markdown plus parsed metadata."""

from __future__ import annotations

from dataclasses import dataclass

from quick_agent.models.agent_spec import AgentSpec


@dataclass
class LoadedAgentFile:
    spec: AgentSpec
    body: str
    step_prompts: dict[str, str]  # prompt_section -> markdown chunk
