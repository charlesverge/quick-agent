from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AgentState:
  memory: dict[str, object]
