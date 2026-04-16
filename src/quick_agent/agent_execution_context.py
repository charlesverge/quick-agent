from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Type

import httpx
import openai
from httpx._config import DEFAULT_LIMITS
from pydantic import BaseModel, JsonValue

from quick_agent.agent_config import AgentConfig
from quick_agent.models.model_spec import ModelSettings, ModelSpec


@dataclass
class AgentExecutionContext:
    config: AgentConfig
    extra_headers: dict[str, str] | None
    extra_body: dict[str, JsonValue] | None
    model_settings_json: ModelSettings
    http_client: httpx.AsyncClient | None
    model_name: str
    client: openai.AsyncOpenAI | None
    effective_base_url: str
    last_run_metrics: dict[str, object] | None = None

    def build_client(self, config: AgentConfig) -> openai.AsyncOpenAI:
        api_key_env = config.model_spec.api_key_env
        api_key = os.environ.get(api_key_env, "noop")
        timeout_seconds = config.model_spec.timeout_seconds
        return config.client or openai.AsyncOpenAI(
            api_key=api_key,
            base_url=config.model_spec.base_url,
            timeout=timeout_seconds,
            http_client=self.http_client,
        )

    @classmethod
    def from_config(
        cls, config: AgentConfig, model_settings_json: ModelSettings | None = None
    ) -> AgentExecutionContext:
        extra_headers = cls._build_extra_headers(config)
        extra_body = cls._build_extra_body(config)
        http_client = (
            config.http_client
            if config.http_client is not None
            else cls._build_http_client(config, extra_headers)
        )
        model_settings_json = (
            model_settings_json
            if model_settings_json is not None
            else cls.build_model_settings(config)
        )
        client = config.client
        if client is None:
            api_key = os.environ.get(config.model_spec.api_key_env, "noop")
            timeout_seconds = config.model_spec.timeout_seconds
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=config.model_spec.base_url,
                timeout=timeout_seconds,
                http_client=http_client,
            )
        effective_base_url = config.model_spec.base_url.rstrip("/")
        return cls(
            config=config,
            extra_headers=extra_headers,
            extra_body=extra_body,
            model_settings_json=model_settings_json,
            http_client=http_client,
            model_name=config.model_spec.model_name,
            client=client,
            effective_base_url=effective_base_url,
        )

    def build_structured_model_settings(
        self, *, schema_cls: Type[BaseModel]
    ) -> ModelSettings:
        model_settings: ModelSettings = self.model_settings_json
        base_url = self.config.model_spec.base_url.rstrip("/")
        if base_url == "https://api.openai.com/v1":
            if model_settings is None:
                model_settings = ModelSettings()
            else:
                model_settings = model_settings.model_copy()
            extra_body_obj = model_settings.extra_body
            extra_body: dict[str, object] = {}
            if isinstance(extra_body_obj, dict):
                extra_body = dict(extra_body_obj)
            schema = schema_cls.model_json_schema()
            self._apply_strict_schema(schema)
            extra_body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_cls.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }
            model_settings.extra_body = extra_body
        return model_settings

    def _apply_strict_schema(self, schema: dict[str, JsonValue]) -> None:
        if "properties" in schema:
            schema["additionalProperties"] = False
            props = schema.get("properties")
            if isinstance(props, dict):
                schema["required"] = list(props.keys())
        defs = schema.get("$defs")
        if isinstance(defs, dict):
            for def_schema in defs.values():
                if isinstance(def_schema, dict):
                    self._apply_strict_schema(def_schema)

    @staticmethod
    def _build_extra_headers(config: AgentConfig) -> dict[str, str] | None:
        headers: dict[str, str] = dict(config.model_spec.extra_headers or {})
        if config.extra_headers is not None:
            headers.update(config.extra_headers)
        return headers if headers else None

    @staticmethod
    def _build_extra_body(config: AgentConfig) -> dict[str, JsonValue] | None:
        extra_body: dict[str, JsonValue] = dict(config.model_spec.extra_body or {})
        if config.extra_body is not None:
            extra_body.update(config.extra_body)
        return extra_body if extra_body else None

    @staticmethod
    def _build_http_client(
        config: AgentConfig, extra_headers: dict[str, str] | None
    ) -> httpx.AsyncClient | None:
        timeout_seconds = config.model_spec.timeout_seconds or 60.0
        keepalive_expiry_seconds = config.model_spec.keepalive_expiry_seconds
        limits: httpx.Limits = DEFAULT_LIMITS
        if keepalive_expiry_seconds is not None:
            limits = httpx.Limits(
                max_connections=100, keepalive_expiry=keepalive_expiry_seconds
            )

        headers = extra_headers if extra_headers else None
        event_hooks: dict[str, list[Callable[..., Any]]] | None = None
        if config.record_http_traffic and config.recorder is not None:
            event_hooks = {
                "request": [config.recorder._record_http_request],
                "response": [config.recorder._record_http_response],
            }

        if (
            timeout_seconds is None
            and limits is None
            and event_hooks is None
            and headers is None
        ):
            return None

        return httpx.AsyncClient(
            timeout=timeout_seconds,
            limits=limits,
            headers=headers,
            event_hooks=event_hooks,
        )

    @classmethod
    def build_model_settings(
        cls,
        config: AgentConfig,
        model_spec: ModelSpec | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, JsonValue] | None = None,
    ) -> ModelSettings:
        if model_spec is None:
            model_spec = config.model_spec
        settings = ModelSettings()
        if extra_headers is None:
            extra_headers = cls._build_extra_headers(config)
        if extra_body is None:
            extra_body = cls._build_extra_body(config)
        if extra_headers:
            settings.extra_headers = extra_headers

        if model_spec.provider == "openai-compatible":
            if model_spec.base_url != "https://api.openai.com/v1":
                extra_body_obj: dict[str, object] = {"format": "json"}
                if extra_body:
                    extra_body_obj.update(extra_body)
                if extra_body_obj:
                    settings.extra_body = extra_body_obj
            elif extra_body:
                extra_body_dict = dict(extra_body)
                options = extra_body_dict.get("options")
                if isinstance(options, dict) and "num_ctx" in options:
                    options = {k: v for k, v in options.items() if k != "num_ctx"}
                    if options:
                        extra_body_dict["options"] = options
                    else:
                        extra_body_dict.pop("options", None)
                settings.extra_body = extra_body_dict

        return settings
