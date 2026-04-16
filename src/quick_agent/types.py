from typing import Any

from pydantic import BaseModel

type AgentResult = BaseModel | dict[str, object] | str | list["AgentResult"]
type StepOutput = BaseModel | str | dict[str, Any]
