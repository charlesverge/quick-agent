from __future__ import annotations

import logging
import os

import httpx
import openai
from pydantic import BaseModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

from quick_agent.models.loaded_agent_file import LoadedAgentFile
from quick_agent.models.model_spec import ModelSpec
from quick_agent.ollama_utils import OllamaSafeChatModel
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
    tool_mode: str = "default",
) -> OpenAIChatModel:
    api_key = os.environ.get(model_spec.api_key_env, "noop")
    provider = (
        OpenAIProvider(openai_client=client)
        if client is not None
        else OpenAIProvider(
            base_url=model_spec.base_url, api_key=api_key, http_client=http_client
        )
    )
    profile = _build_model_profile(tool_mode)
    logger.info(f"build_model {tool_mode}")

    model_cls: type[OpenAIChatModel] = OpenAIChatModel
    if tool_mode in ("with_tools", "no_tools"):
        model_cls = OllamaSafeChatModel
    if profile is not None:
        return model_cls(model_spec.model_name, provider=provider, profile=profile)
    return model_cls(model_spec.model_name, provider=provider)


def _build_model_profile(
    tool_mode: str,
) -> OpenAIModelProfile | None:
    """Build an OpenAIModelProfile based on the tool_mode setting.

    - default: no custom profile (pydantic_ai defaults).
    - no_tools: prompted structured output, avoids tool calling entirely.
    - with_tools: standard tool mode with OllamaSafeChatModel subclass.
    - prompted_tools: prompted structured output with tools (experimental).
    """
    if tool_mode in ("no_tools", "prompted_tools"):
        return OpenAIModelProfile(
            openai_supports_strict_tool_definition=False,
            default_structured_output_mode="prompted",
            supports_json_object_output=True,
        )
    if tool_mode == "with_tools":
        return OpenAIModelProfile(
            openai_supports_strict_tool_definition=False,
        )
    return None
