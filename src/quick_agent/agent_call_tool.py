"""Callable tool for inter-agent calls."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from quick_agent.input_adaptors import InputAdaptor, TextInput


class AgentCallTool:
    def __init__(
        self,
        call_agent: Callable[[str, InputAdaptor | Path], Awaitable[BaseModel | str]],
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

    async def __call__(
        self,
        agent: str,
        input_file: str | None = None,
        input_text: str | None = None,
    ) -> dict[str, Any]:
        """
        Call another agent by ID with an input file path or inline text.
        Returns JSON-serializable dict output if structured, else {"text": "..."}.
        """
        if input_file and input_text:
            raise ValueError("Provide only one of input_file or input_text.")
        if not input_file and input_text is None:
            raise ValueError("Provide either input_file or input_text.")
        if input_text is not None:
            input_data: InputAdaptor | Path = TextInput(input_text)
        else:
            if input_file is None:
                raise ValueError("Provide either input_file or input_text.")
            resolved_input = self._resolve_input_file(input_file)
            input_data = resolved_input
        out = await self._call_agent(agent, input_data)
        if isinstance(out, BaseModel):
            return out.model_dump()
        return {"text": out}
