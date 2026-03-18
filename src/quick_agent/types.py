from pydantic import BaseModel

type AgentResult = BaseModel | dict[str, object] | str
