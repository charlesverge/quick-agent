"""Callable tool for inter-agent calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel


class AgentCallTool:
    def __init__(
        self,
        call_agent: Callable[[str, Path], Awaitable[BaseModel | str]],
        run_input_source_path: str,
    ) -> None:
        self._call_agent = call_agent
        self._run_input_source_path = run_input_source_path
        self.__name__ = "agent_call"

    def _resolve_input_file(self, raw_input_file: str) -> Path:
        cleaned = raw_input_file.strip()
        if (
            (cleaned.startswith("\"") and cleaned.endswith("\""))
            or (cleaned.startswith("'") and cleaned.endswith("'"))
        ):
            cleaned = cleaned[1:-1]
        base_dir = Path(self._run_input_source_path).parent
        cleaned = cleaned.replace("{base_directory}", str(base_dir))
        path = Path(cleaned)
        if not path.is_absolute():
            path = base_dir / path
        return path

    async def __call__(self, agent: str, input_file: str) -> dict[str, Any]:
        """
        Call another agent by ID with an input file path.
        Returns JSON-serializable dict output if structured, else {"text": "..."}.
        """
        resolved_input = self._resolve_input_file(input_file)
        out = await self._call_agent(agent, resolved_input)
        if isinstance(out, BaseModel):
            return out.model_dump()
        return {"text": out}
