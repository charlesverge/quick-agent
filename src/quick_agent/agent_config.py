from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import httpx
import openai
from pydantic import JsonValue
from pydantic_ai.toolsets import FunctionToolset

from quick_agent.models.batch_request import BatchImportOutcome
from quick_agent.models.loaded_agent_file import LoadedAgentFile
from quick_agent.models.model_spec import ModelSpec
from quick_agent.models.run_input import RunInput
from quick_agent.recorder import Recorder


@dataclass
class AgentConfig:
    agent_id: str
    toolset: FunctionToolset[Any] | None
    tool_ids: list[str]
    memory: dict[str, Any]
    model_spec: ModelSpec
    client: openai.AsyncOpenAI | None
    http_client: httpx.AsyncClient | None
    tool_mode: str
    extra_headers: dict[str, str] | None
    extra_body: dict[str, JsonValue] | None
    record_http_traffic: bool
    run_input: RunInput
    loaded: LoadedAgentFile
    extra_tools: list[str] | None
    recorder: Recorder | None
    state: object
    import_outcome: Callable[..., BatchImportOutcome]
    batch_call: Callable[..., object] | None = None
