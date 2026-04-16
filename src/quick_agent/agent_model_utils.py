from __future__ import annotations

import logging
import os

import httpx
import openai
from pydantic import BaseModel

from quick_agent.models.loaded_agent_file import LoadedAgentFile
from quick_agent.models.model_spec import ModelSpec
from quick_agent.tools_loader import import_symbol

logger = logging.getLogger(__name__)


def resolve_schema(loaded: LoadedAgentFile, schema_name: str) -> type[BaseModel]:
    if schema_name not in loaded.spec.schemas:
        raise KeyError(f"Schema {schema_name!r} not registered in agent.md schemas.")
    cls = import_symbol(loaded.spec.schemas[schema_name])
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        raise TypeError(
            f"Schema {schema_name!r} must be a Pydantic BaseModel subclass."
        )
    return cls


def build_model(
    model_spec: ModelSpec,
    *,
    http_client: httpx.AsyncClient | None = None,
    client: openai.AsyncOpenAI | None = None,
) -> openai.AsyncOpenAI:
    """Build an AsyncOpenAI client for the given model spec."""
    api_key = os.environ.get(model_spec.api_key_env, "noop")
    if client is not None:
        return client
    return openai.AsyncOpenAI(
        api_key=api_key,
        base_url=model_spec.base_url,
        http_client=http_client,
    )
