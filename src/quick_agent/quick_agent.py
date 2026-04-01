"""Agent execution logic."""

from __future__ import annotations

import json
import logging
import os
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Type, TypeAlias, TypedDict

import httpx
import openai
from httpx._config import DEFAULT_LIMITS
from openai.types import chat
from pydantic import BaseModel, JsonValue, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.exceptions import (
    QuickAgentChatNotSupportedException,
    QuickAgentException,
    QuickAgentToolsNotSupportedException,
)
from quick_agent.input_adaptors import FileInput, InputAdaptor, TextInput
from quick_agent.io_utils import write_output
from quick_agent.json_utils import extract_first_json_object
from quick_agent.mapping.map_chunks import MapChunks
from quick_agent.mapping.map_paragraphs import MapParagraphs
from quick_agent.models.chain_step_spec import ChainStepSpec
from quick_agent.models.content_processing_spec import ChunkProcessingSpec
from quick_agent.models.loaded_agent_file import LoadedAgentFile
from quick_agent.models.model_spec import ModelSpec
from quick_agent.models.run_input import RunInput
from quick_agent.prompting import make_user_prompt
from quick_agent.samplers.simple_ratios import SampleRatios
from quick_agent.single_shot import run_single_shot
from quick_agent.tools_loader import import_symbol
from quick_agent.types import AgentResult

logger = logging.getLogger(__name__)


StepOutput: TypeAlias = str | dict[str, Any]


class ChainState(TypedDict):
    agent_id: str
    steps: dict[str, StepOutput]
    last_step_output: StepOutput | None


class ExecutionLogEntry:
    def __init__(self, *, request_context: dict[str, object], call_site: str) -> None:
        self.request_context = request_context
        self.call_site = call_site

    def _request_from_context(self) -> dict[str, object] | None:
        request_obj = self.request_context.get("request")
        if isinstance(request_obj, dict):
            return request_obj
        return None

    def _reconstructed_request_from_context(self) -> dict[str, object] | None:
        base_url_obj = self.request_context.get("base_url")
        model_name_obj = self.request_context.get("model_name")
        user_prompt_obj = self.request_context.get("user_prompt")
        system_prompt_obj = self.request_context.get("system_prompt")
        instructions_obj = self.request_context.get("instructions")
        model_settings_obj = self.request_context.get("model_settings")
        if (
            not isinstance(base_url_obj, str)
            or not isinstance(model_name_obj, str)
            or not isinstance(user_prompt_obj, str)
        ):
            return None
        messages: list[dict[str, str]] = []
        system_parts: list[str] = []
        if isinstance(system_prompt_obj, str) and system_prompt_obj:
            system_parts.append(system_prompt_obj)
        elif isinstance(system_prompt_obj, list):
            for item in system_prompt_obj:
                if isinstance(item, str) and item:
                    system_parts.append(item)
        if isinstance(instructions_obj, str) and instructions_obj:
            system_parts.append(instructions_obj)
        if system_parts:
            messages.append({"role": "system", "content": "\n".join(system_parts)})
        messages.append({"role": "user", "content": user_prompt_obj})
        body: dict[str, object] = {"model": model_name_obj, "messages": messages}
        if isinstance(model_settings_obj, dict):
            extra_body_obj = model_settings_obj.get("extra_body")
            if isinstance(extra_body_obj, dict):
                for key, value in extra_body_obj.items():
                    if key not in body:
                        body[key] = value
        base_url = base_url_obj.rstrip("/")
        if base_url.endswith("/chat/completions"):
            url = base_url
        else:
            url = f"{base_url}/chat/completions"
        return {
            "method": "POST",
            "url": url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, ensure_ascii=False),
        }

    def to_curl(self) -> str:
        request_obj = self._request_from_context()
        if request_obj is None:
            request_obj = self._reconstructed_request_from_context()
        if request_obj is None:
            return "curl"
        method_obj = request_obj.get("method")
        url_obj = request_obj.get("url")
        headers_obj = request_obj.get("headers")
        body_obj = request_obj.get("body")
        if not isinstance(method_obj, str) or not isinstance(url_obj, str):
            return "curl"
        command_parts: list[str] = ["curl", "-X", shlex.quote(method_obj)]
        if isinstance(headers_obj, dict):
            for key_obj, value_obj in headers_obj.items():
                if not isinstance(key_obj, str) or not isinstance(value_obj, str):
                    continue
                header_value = f"{key_obj}: {value_obj}"
                command_parts.extend(["-H", shlex.quote(header_value)])
        if isinstance(body_obj, str) and body_obj:
            command_parts.extend(["--data-raw", shlex.quote(body_obj)])
        command_parts.append(shlex.quote(url_obj))
        return " ".join(command_parts)


class QuickAgent:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        tools: AgentTools,
        directory_permissions: DirectoryPermissions,
        agent_id: str,
        input_data: InputAdaptor | Path,
        extra_tools: list[str] | None,
        model: ModelSpec | None = None,
        write_output: bool = True,
        record_http_traffic: bool = False,
        enable_llm_request_logging: bool = False,
        llm_log_path: Path | str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, JsonValue] | None = None,
        client: openai.AsyncOpenAI | None = None,
    ) -> None:
        self._registry: AgentRegistry = registry
        self._tools: AgentTools = tools
        self._directory_permissions: DirectoryPermissions = directory_permissions
        self._agent_id: str = agent_id
        self._input_data: InputAdaptor | Path = input_data
        self._extra_tools: list[str] | None = extra_tools
        self.loaded: LoadedAgentFile = self._registry.get(self._agent_id)
        output_file = self.loaded.spec.output.file
        self._write_output_file: bool = write_output and bool(output_file)
        safe_dir = self.loaded.spec.safe_dir
        if safe_dir is not None and Path(safe_dir).is_absolute():
            raise ValueError("safe_dir must be a relative path.")
        self.permissions: DirectoryPermissions = self._directory_permissions.scoped(
            safe_dir
        )
        if isinstance(self._input_data, InputAdaptor):
            input_adaptor = self._input_data
        else:
            input_adaptor = FileInput(self._input_data, self.permissions)
        self.run_input: RunInput = input_adaptor.load()

        self.tool_ids: list[str] = self._build_tool_ids()
        self.toolset: FunctionToolset[Any] | None = self._build_toolset()
        self.model_spec: ModelSpec = model or self.loaded.spec.model
        self._record_http_traffic: bool = record_http_traffic
        self._http_traffic_entries: list[dict[str, object]] = []
        self.http_request_log: list[dict[str, object]] = []
        self.http_response_log: list[dict[str, object]] = []
        self._http_log_max_entries: int = 200
        self.execution_log: list[ExecutionLogEntry] = []

        headers: dict[str, str] = dict(self.model_spec.extra_headers or {})
        if extra_headers is not None:
            headers.update(extra_headers)
        self.extra_headers = headers

        new_extra_body: dict[str, JsonValue] = dict(self.model_spec.extra_body or {})
        if extra_body is not None:
            new_extra_body.update(extra_body)
        self.extra_body: dict[str, JsonValue] = new_extra_body
        self.client: openai.AsyncOpenAI | None = client

        self._http_client: httpx.AsyncClient | None = self._build_http_client()
        self.tool_mode: str = self.loaded.spec.tool_mode
        logger.info(f"Initialized QuickAgent {self._agent_id}, tool_mode: {self.tool_mode}")
        self.model: OpenAIChatModel = build_model(
            self.model_spec,
            http_client=self._http_client,
            client=self.client,
            tool_mode=self.tool_mode,
        )
        self.state: ChainState = self._init_state()
        self._enable_llm_request_logging: bool = enable_llm_request_logging
        if llm_log_path is None:
            llm_log_path = Path("log/results.log")
        self._llm_log_path: Path = Path(llm_log_path)
        self.model_settings_json: ModelSettings | None = self._build_model_settings(
            self.model_spec
        )
        self.last_run_metrics: dict[str, object] | None = None

    async def run(self) -> AgentResult:
        self.last_run_metrics = None
        if self.has_tools():
            if self.toolset is None:
                raise ValueError("Toolset is missing while tools are enabled.")
            self._tools.maybe_inject_agent_call(
                self.tool_ids,
                self.toolset,
                self.run_input.source_path,
                self._run_nested_agent,
            )
        self._apply_sample_processing()
        chunk_output = await self._apply_chunk_processing()
        if chunk_output is not None:
            if self._write_output_file:
                self._write_last_step_output(chunk_output)
            handoff_output = await self._handle_handoff(chunk_output)
            if handoff_output is not None:
                return handoff_output
            return chunk_output
        if self._is_empty_agent_body():
            output: AgentResult = self.run_input.text
            if self._write_output_file:
                self._write_last_step_output(output)
            handoff_output = await self._handle_handoff(output)
            if handoff_output is not None:
                return handoff_output
            return output

        try:
            last_step_output = await self._run_chain()

            final_output: AgentResult = last_step_output
            if self.loaded.spec.output.return_compiled_output:
                final_output = self._compiled_output(last_step_output)

            if self._write_output_file:
                self._write_last_step_output(final_output)

            handoff_output = await self._handle_handoff(last_step_output)
            if handoff_output is not None:
                return handoff_output

            return final_output
        finally:
            self._write_llm_request_log(None)

    def _apply_sample_processing(self) -> None:
        content_processing = self.loaded.spec.content_processing
        if content_processing is None or content_processing.sample is None:
            return
        sample_result = SampleRatios().run(
            self.run_input.text, content_processing.sample
        )
        self.run_input = self.run_input.model_copy(update={"text": sample_result})
        debug_output_file = content_processing.sample.debug_output_file
        if debug_output_file:
            write_output(Path(debug_output_file), sample_result, self.permissions)

    async def _apply_chunk_processing(self) -> list[AgentResult] | None:
        content_processing = self.loaded.spec.content_processing
        if content_processing is None or content_processing.chunk_processing is None:
            return None
        map_config = content_processing.chunk_processing
        chunk_texts = self._run_chunk_processing(self.run_input.text, map_config)
        if self._is_empty_agent_body():
            empty_items: list[AgentResult] = []
            empty_items.extend(chunk_texts)
            return empty_items
        items: list[AgentResult] = []
        index = 0
        while index < len(chunk_texts):
            chunk_text = chunk_texts[index]
            chunk_output = await self._run_chunk_agent(chunk_text)
            if isinstance(chunk_output, BaseModel):
                items.append(chunk_output.model_dump())
            else:
                items.append(chunk_output)
            index += 1
        return items

    async def _run_chunk_agent(self, chunk_text: str) -> AgentResult:
        chunk_agent = QuickAgent(
            registry=self._registry,
            tools=self._tools,
            directory_permissions=self._directory_permissions,
            agent_id=self._agent_id,
            input_data=TextInput(chunk_text),
            extra_tools=self._extra_tools,
            model=self.model_spec,
            write_output=False,
            record_http_traffic=self._record_http_traffic,
            enable_llm_request_logging=self._enable_llm_request_logging,
            llm_log_path=self._llm_log_path,
            extra_headers=self.extra_headers,
            extra_body=self.extra_body,
            client=self.client,
        )
        if chunk_agent.has_tools():
            if chunk_agent.toolset is None:
                raise ValueError("Toolset is missing while tools are enabled.")
            chunk_agent._tools.maybe_inject_agent_call(
                chunk_agent.tool_ids,
                chunk_agent.toolset,
                chunk_agent.run_input.source_path,
                chunk_agent._run_nested_agent,
            )
        last_step_output = await chunk_agent._run_chain()
        if chunk_agent.loaded.spec.output.return_compiled_output:
            return chunk_agent._compiled_output(last_step_output)
        return last_step_output

    def _run_chunk_processing(
        self, text: str, map_config: ChunkProcessingSpec
    ) -> list[str]:
        if map_config.provider != "semchunks":
            raise ValueError("chunk_processing.provider must be 'semchunks'.")
        if map_config.mode == "map_chunks":
            return MapChunks().run(text, map_config)
        if map_config.mode == "map_paragraphs":
            return MapParagraphs().run(text, map_config)
        raise ValueError(
            "chunk_processing.mode must be 'map_chunks' or 'map_paragraphs'."
        )

    def _is_empty_agent_body(self) -> bool:
        if self.loaded.spec.chain:
            return False
        if self.loaded.instructions.strip():
            return False
        if isinstance(self.loaded.system_prompt, list):
            index = 0
            while index < len(self.loaded.system_prompt):
                if self.loaded.system_prompt[index].strip():
                    return False
                index += 1
            return True
        return not self.loaded.system_prompt.strip()

    async def _run_nested_agent(
        self, agent_id: str, input_data: InputAdaptor | Path
    ) -> AgentResult:
        nested_write_output = self.loaded.spec.nested_output == "file"
        agent = QuickAgent(
            registry=self._registry,
            tools=self._tools,
            directory_permissions=self._directory_permissions,
            agent_id=agent_id,
            input_data=input_data,
            extra_tools=None,
            model=self.model_spec,
            write_output=nested_write_output,
            record_http_traffic=self._record_http_traffic,
            enable_llm_request_logging=self._enable_llm_request_logging,
            llm_log_path=self._llm_log_path,
            client=self.client,
        )
        return await agent.run()

    def _init_state(self) -> ChainState:
        return {
            "agent_id": self._agent_id,
            "steps": {},
            "last_step_output": None,
        }

    def _build_model_settings(self, model_spec: ModelSpec) -> ModelSettings | None:
        settings: ModelSettings = {}
        if self.extra_headers:
            settings["extra_headers"] = self.extra_headers

        if model_spec.provider == "openai-compatible":
            # Ollama OpenAI-compatible API uses "format": "json" to force JSON output.
            if model_spec.base_url != "https://api.openai.com/v1":
                extra_body: dict = {"format": "json"}
                if self.extra_body:
                    extra_body.update(self.extra_body)
                if extra_body:
                    settings["extra_body"] = extra_body
            elif self.extra_body:
                extra_body = dict(self.extra_body)
                options = extra_body.get("options")
                if isinstance(options, dict) and "num_ctx" in options:
                    options = {k: v for k, v in options.items() if k != "num_ctx"}
                    if options:
                        extra_body["options"] = options
                    else:
                        extra_body.pop("options", None)
                if extra_body:
                    settings["extra_body"] = extra_body

        if not settings:
            return None

        return settings

    def _build_http_client(self) -> httpx.AsyncClient | None:
        timeout_seconds = self.model_spec.timeout_seconds or 60.0
        keepalive_expiry_seconds = self.model_spec.keepalive_expiry_seconds
        limits: httpx.Limits = DEFAULT_LIMITS
        if keepalive_expiry_seconds is not None:
            limits = httpx.Limits(
                max_connections=100, keepalive_expiry=keepalive_expiry_seconds
            )

        headers = self.extra_headers if self.extra_headers else None
        event_hooks: dict[str, list[Callable[..., Any]]] | None = None
        if self._record_http_traffic:
            event_hooks = {
                "request": [self._record_http_request],
                "response": [self._record_http_response],
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

    def _build_structured_model_settings(
        self, *, schema_cls: Type[BaseModel]
    ) -> ModelSettings | None:
        model_settings: ModelSettings | None = self.model_settings_json
        provider = getattr(self.model, "provider", None)
        base_url = getattr(provider, "base_url", None)
        if base_url == "https://api.openai.com/v1":
            if self.model_settings_json is None:
                model_settings_dict: ModelSettings = {}
            else:
                model_settings_dict = self.model_settings_json
            extra_body_obj = model_settings_dict.get("extra_body")
            extra_body: dict = {}
            if isinstance(extra_body_obj, dict):
                extra_body = dict(extra_body_obj)
            if "response_format" not in extra_body:
                extra_body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_cls.__name__,
                        "schema": schema_cls.model_json_schema(),
                        "strict": True,
                    },
                }
            model_settings_dict["extra_body"] = extra_body
            model_settings = model_settings_dict
        return model_settings

    async def _run_step(
        self,
        *,
        step: ChainStepSpec,
    ) -> str | BaseModel:
        if step.kind == "text":
            return await self._run_text_step(
                step=step,
            )

        if step.kind == "structured":
            return await self._run_structured_step(
                step=step,
            )

        raise NotImplementedError(f"Unknown step kind: {step.kind}")

    def _build_step_instructions(self, step_prompt: str) -> str:
        if not self.loaded.instructions:
            return step_prompt
        return f"{self.loaded.instructions}{step_prompt}"

    def _build_single_shot_prompt(self) -> str:
        return make_user_prompt(self.run_input, self.state)

    def _json_compatible_value(self, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            converted: dict[str, object] = {}
            for key, item in value.items():
                converted[str(key)] = self._json_compatible_value(item)
            return converted
        if isinstance(value, list):
            return [self._json_compatible_value(item) for item in value]
        if isinstance(value, tuple):
            return [self._json_compatible_value(item) for item in value]
        return str(value)

    def _normalize_usage_metrics(self, usage: object) -> dict[str, object]:
        usage_dict: dict[str, object] = {}
        if isinstance(usage, dict):
            for key, value in usage.items():
                usage_dict[str(key)] = self._json_compatible_value(value)
            return usage_dict
        if isinstance(usage, BaseModel):
            payload = usage.model_dump(exclude_none=True)
            if isinstance(payload, dict):
                for key, value in payload.items():
                    usage_dict[str(key)] = self._json_compatible_value(value)
            return usage_dict
        model_dump = getattr(usage, "model_dump", None)
        if callable(model_dump):
            payload = model_dump(exclude_none=True)
            if isinstance(payload, dict):
                for key, value in payload.items():
                    usage_dict[str(key)] = self._json_compatible_value(value)
                return usage_dict
        return usage_dict

    def _extract_finish_reason(self, response: object | None) -> str | None:
        if response is None:
            return None
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            return None
        first_choice = choices[0]
        finish_reason = getattr(first_choice, "finish_reason", None)
        if isinstance(finish_reason, str) and finish_reason:
            return finish_reason
        return None

    def _capture_metrics(self, *, usage: object, response: object | None) -> None:
        model = self.model_spec.model_name
        if response is not None:
            response_model = getattr(response, "model", None)
            if isinstance(response_model, str) and response_model:
                model = response_model
        metrics: dict[str, object] = {
            "provider": self.model_spec.provider,
            "model": model,
            "usage": self._normalize_usage_metrics(usage),
        }
        if response is not None:
            completion_id = getattr(response, "id", None)
            if isinstance(completion_id, str) and completion_id:
                metrics["completion_id"] = completion_id
            created = getattr(response, "created", None)
            if isinstance(created, (int, float, str)):
                metrics["created"] = created
            system_fingerprint = getattr(response, "system_fingerprint", None)
            if isinstance(system_fingerprint, str) and system_fingerprint:
                metrics["system_fingerprint"] = system_fingerprint
            finish_reason = self._extract_finish_reason(response)
            if finish_reason is not None:
                metrics["finish_reason"] = finish_reason
        self.last_run_metrics = metrics

    def _capture_pydantic_ai_metrics(self, result: object) -> None:
        usage: object = {}
        response = getattr(result, "response", None)
        usage_getter = getattr(result, "usage", None)
        if callable(usage_getter):
            usage = usage_getter()
        self._capture_metrics(usage=usage, response=response)

    def _capture_openai_sdk_metrics(self, response: object) -> None:
        usage = getattr(response, "usage", {})
        self._capture_metrics(usage=usage, response=response)

    def _effective_base_url(self) -> str:
        if self.client is not None:
            return str(self.client.base_url).rstrip("/")
        return self.model_spec.base_url.rstrip("/")

    def _record_llm_request(
        self,
        *,
        call_site: str,
        step_id: str | None,
        step_kind: str,
        output_schema: str | None,
        instructions: str | None,
        system_prompt: str | list[str],
        user_prompt: str,
        model_settings: ModelSettings | None,
    ) -> None:
        self._record_execution_log(
            call_site=call_site,
            instructions=instructions,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_settings=model_settings,
        )
        payload: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_state": "before_request_start",
            "agent_id": self._agent_id,
            "model": {
                "provider": self.model_spec.provider,
                "base_url": self._effective_base_url(),
                "model_name": self.model_spec.model_name,
            },
            "step": {
                "id": step_id,
                "kind": step_kind,
                "output_schema": output_schema,
            },
            "call_site": call_site,
            "system_prompt": system_prompt,
            "instructions": instructions,
            "user_prompt": user_prompt,
            "model_settings": self._json_compatible_value(model_settings),
            "tool_ids": self.tool_ids,
        }
        self._write_llm_request_log(payload)

    def _write_llm_request_log(self, payload: dict[str, object] | None) -> None:
        prefix = "QuickAgent._write_llm_request_log"
        if not self._enable_llm_request_logging or payload is None:
            return
        try:
            self._llm_log_path.parent.mkdir(parents=True, exist_ok=True)
            entry = json.dumps(payload, indent=2)
            with self._llm_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write("[LLM_REQUEST]\n")
                log_file.write(entry)
                log_file.write("\n\n")
        except OSError:
            logger.exception(
                "%s: file=%s > Failed to write LLM request log",
                prefix,
                self._llm_log_path,
            )

    def _map_model_http_error(
        self, error: ModelHTTPError
    ) -> QuickAgentException | None:
        body = error.body
        message = ""
        if isinstance(body, dict):
            body_message = body.get("message")
            if isinstance(body_message, str):
                message = body_message
        elif isinstance(body, str):
            message = body
        if "does not support tools" in message:
            return QuickAgentToolsNotSupportedException(
                model_name=error.model_name, message=message
            )
        if "does not support chat" in message:
            return QuickAgentChatNotSupportedException(
                model_name=error.model_name, message=message
            )
        return None

    def _decode_http_bytes(self, value: bytes) -> str:
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")

    def _record_http_traffic_entry(self, entry: dict[str, object]) -> None:
        self._http_traffic_entries.append(entry)
        if len(self._http_traffic_entries) > self._http_log_max_entries:
            del self._http_traffic_entries[0]

    def _record_http_request_entry(self, request_entry: dict[str, object]) -> None:
        self.http_request_log.append(request_entry)
        if len(self.http_request_log) > self._http_log_max_entries:
            del self.http_request_log[0]

    def _record_http_response_entry(self, response_entry: dict[str, object]) -> None:
        self.http_response_log.append(response_entry)
        if len(self.http_response_log) > self._http_log_max_entries:
            del self.http_response_log[0]

    async def _record_http_request(self, request: httpx.Request) -> None:
        request_body: str | None = None
        if request.content:
            request_body = self._decode_http_bytes(request.content)
        entry: dict[str, object] = {
            "event": "request",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request": {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": request_body,
            },
        }
        request_obj = entry.get("request")
        if isinstance(request_obj, dict):
            self._record_http_request_entry(request_obj)
        self._record_http_traffic_entry(entry)

    async def _record_http_response(self, response: httpx.Response) -> None:
        response_body: str | None = None
        response_content = await response.aread()
        if response_content:
            response_body = self._decode_http_bytes(response_content)
        request_body: str | None = None
        if response.request.content:
            request_body = self._decode_http_bytes(response.request.content)
        entry: dict[str, object] = {
            "event": "response",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request": {
                "method": response.request.method,
                "url": str(response.request.url),
                "headers": dict(response.request.headers),
                "body": request_body,
            },
            "response": {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response_body,
            },
        }
        response_obj = entry.get("response")
        if isinstance(response_obj, dict):
            self._record_http_response_entry(response_obj)
        self._record_http_traffic_entry(entry)

    def _last_http_exchange_context(self) -> dict[str, object]:
        if self.http_request_log:
            context: dict[str, object] = {
                "request": self.http_request_log[-1],
                "request_source": "quick_agent_http_traffic_log",
            }
            if self.http_response_log:
                context["response"] = self.http_response_log[-1]
            return context
        for entry in reversed(self._http_traffic_entries):
            if entry.get("event") == "response":
                request_obj = entry.get("request")
                response_obj = entry.get("response")
                if isinstance(request_obj, dict):
                    exchange_context: dict[str, object] = {
                        "request": request_obj,
                        "request_source": "quick_agent_http_traffic_log",
                    }
                    if isinstance(response_obj, dict):
                        exchange_context["response"] = response_obj
                    return exchange_context
        for entry in reversed(self._http_traffic_entries):
            if entry.get("event") == "request":
                request_obj = entry.get("request")
                if isinstance(request_obj, dict):
                    return {
                        "request": request_obj,
                        "request_source": "quick_agent_http_traffic_log",
                    }
        return {}

    def _unexpected_model_behavior_request_context(
        self,
        *,
        instructions: str | None,
        system_prompt: str | list[str],
        user_prompt: str,
        model_settings: ModelSettings | None,
    ) -> dict[str, object]:
        context: dict[str, object] = {
            "base_url": self._effective_base_url(),
            "model_name": self.model_spec.model_name,
            "instructions": instructions,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model_settings": self._json_compatible_value(model_settings),
        }
        context.update(self._last_http_exchange_context())
        return context

    def _record_execution_log(
        self,
        *,
        call_site: str,
        instructions: str | None,
        system_prompt: str | list[str],
        user_prompt: str,
        model_settings: ModelSettings | None,
    ) -> dict[str, object]:
        request_context = self._unexpected_model_behavior_request_context(
            instructions=instructions,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_settings=model_settings,
        )
        self.execution_log.append(
            ExecutionLogEntry(request_context=request_context, call_site=call_site)
        )
        if len(self.execution_log) > self._http_log_max_entries:
            del self.execution_log[0]
        return request_context

    async def _run_text_step(
        self,
        *,
        step: ChainStepSpec,
    ) -> str:
        prefix = "QuickAgent._run_text_step"
        user_prompt = make_user_prompt(self.run_input, self.state)
        step_prompt = self.loaded.step_prompts[step.prompt_section]
        step_instructions = self._build_step_instructions(step_prompt)
        toolsets = self._toolsets_for_run()
        agent = Agent(
            self.model,
            instructions=step_instructions,
            system_prompt=self.loaded.system_prompt,
            toolsets=toolsets,
            output_type=str,
            model_settings=self.model_settings_json,
        )
        self._record_llm_request(
            call_site="run_text_step",
            step_id=step.id,
            step_kind=step.kind,
            output_schema=step.output_schema,
            instructions=step_instructions,
            system_prompt=self.loaded.system_prompt,
            user_prompt=user_prompt,
            model_settings=self.model_settings_json,
        )
        logger.info(
            "%s: model=%s step=%s > Calling model",
            prefix,
            self.model_spec.model_name,
            step.id,
        )
        try:
            result = await agent.run(user_prompt)
        except ModelHTTPError as error:
            mapped_error = self._map_model_http_error(error)
            if mapped_error is not None:
                raise mapped_error from error
            raise error
        self._capture_pydantic_ai_metrics(result)
        return result.output

    async def _run_single_shot(self) -> BaseModel | str:
        prefix = "QuickAgent._run_single_shot"
        schema_name = self.loaded.spec.output.output_schema
        schema_cls: Type[BaseModel] | None = None
        if schema_name:
            schema_cls = resolve_schema(self.loaded, schema_name)
        logger.info("%s: model=%s > Calling model", prefix, self.model_spec.model_name)
        return await run_single_shot(self, schema_cls=schema_cls)

    async def _run_structured_step(
        self,
        *,
        step: ChainStepSpec,
    ) -> BaseModel:
        prefix = "QuickAgent._run_structured_step"
        if not step.output_schema:
            raise ValueError(f"Step {step.id} is structured but missing output_schema.")
        schema_cls = resolve_schema(self.loaded, step.output_schema)

        model_settings = self._build_structured_model_settings(schema_cls=schema_cls)

        user_prompt = make_user_prompt(self.run_input, self.state)
        step_prompt = self.loaded.step_prompts[step.prompt_section]
        step_instructions = self._build_step_instructions(step_prompt)
        model_settings = self._build_structured_model_settings(schema_cls=schema_cls)
        toolsets = self._toolsets_for_run()
        agent = Agent(
            self.model,
            instructions=step_instructions,
            system_prompt=self.loaded.system_prompt,
            toolsets=toolsets,
            output_type=schema_cls,
            model_settings=model_settings,
        )
        self._record_llm_request(
            call_site="run_structured_step",
            step_id=step.id,
            step_kind=step.kind,
            output_schema=step.output_schema,
            instructions=step_instructions,
            system_prompt=self.loaded.system_prompt,
            user_prompt=user_prompt,
            model_settings=model_settings,
        )
        logger.info(
            "%s: model=%s step=%s schema=%s > Calling model",
            prefix,
            self.model_spec.model_name,
            step.id,
            step.output_schema,
        )
        try:
            result = await agent.run(user_prompt)
        except ModelHTTPError as error:
            mapped_error = self._map_model_http_error(error)
            if mapped_error is not None:
                raise mapped_error from error
            raise error
        self._capture_pydantic_ai_metrics(result)
        raw_output = result.output
        if isinstance(raw_output, BaseModel):
            parsed = raw_output
        elif isinstance(raw_output, dict):
            parsed = schema_cls.model_validate(raw_output)
        else:
            try:
                parsed = schema_cls.model_validate_json(raw_output)
            except ValidationError:
                extracted = extract_first_json_object(raw_output)
                parsed = schema_cls.model_validate_json(extracted)
        return parsed

    async def _run_chain(
        self,
    ) -> BaseModel | str:
        if not self.loaded.spec.chain:
            return await self._run_single_shot()
        last_step_output: BaseModel | str = ""
        for step in self.loaded.spec.chain:
            step_result = await self._run_step(
                step=step,
            )
            if isinstance(step_result, BaseModel):
                step_out: StepOutput = step_result.model_dump()
                self.state["steps"][step.id] = step_out
                self.state["last_step_output"] = step_out
                last_step_output = step_result
            else:
                self.state["steps"][step.id] = step_result
                self.state["last_step_output"] = step_result
                last_step_output = step_result
        return last_step_output

    def _compiled_output(self, last_step_output: BaseModel | str) -> AgentResult:
        # When enabled, return a combined view of all step outputs instead of the last step output.
        if not self.loaded.spec.chain:
            return last_step_output

        fmt = self.loaded.spec.output.format
        if fmt == "json":
            return self._compiled_json_output()
        if fmt == "structured":
            return self._compiled_structured_output()
        return self._compiled_text_output()

    def _compiled_json_output(self) -> dict[str, object]:
        return {
            **self.state.get("steps", {}),
            "last_step_output": self.state.get("last_step_output"),
        }

    def _compiled_text_output(self) -> str:
        values = []
        for step_output in self.state.get("steps", {}).values():
            if isinstance(step_output, dict):
                values.append(json.dumps(step_output, ensure_ascii=False))
            else:
                values.append(str(step_output))
        return "\n".join(values)

    def _compiled_structured_output(self) -> BaseModel:
        compiled_schema = self.loaded.spec.output.compiled_schema
        if not compiled_schema:
            raise ValueError("output.schema must be set for structured compiled output")
        schema_cls = resolve_schema(self.loaded, compiled_schema)

        if not hasattr(schema_cls, "model_fields"):
            raise RuntimeError(
                "Compiled structured output requires a Pydantic model with `model_fields`."
            )
        fields = set(schema_cls.model_fields.keys())

        payload: dict[str, object] = {}
        steps = self.state.get("steps", {})
        for field in fields:
            if field == "last_step_output":
                payload[field] = self.state.get("last_step_output")
            else:
                payload[field] = steps.get(field)

        return schema_cls.model_validate(payload)

    def has_tools(self) -> bool:
        if not self.tool_ids:
            return False
        return True

    def _build_tool_ids(self) -> list[str]:
        if not self.loaded.spec.tools:
            return []
        return list(
            dict.fromkeys((self.loaded.spec.tools or []) + (self._extra_tools or []))
        )

    def _build_toolset(self) -> FunctionToolset[Any] | None:
        if not self.has_tools():
            return None
        return self._tools.build_toolset(self.tool_ids, self.permissions)

    def _toolsets_for_run(self) -> list[FunctionToolset[Any]]:
        if not self.has_tools():
            return []
        toolset = self.toolset
        if toolset is None:
            return []
        return [toolset]

    def _write_last_step_output(self, last_step_output: AgentResult) -> Path:
        output_file = self.loaded.spec.output.file
        if not output_file:
            raise ValueError("Output file is not configured.")
        out_path = Path(output_file)
        if isinstance(last_step_output, BaseModel):
            text = last_step_output.model_dump_json(indent=2)
        elif isinstance(last_step_output, (dict, list)):
            text = json.dumps(last_step_output, indent=2)
        else:
            text = str(last_step_output)
        write_output(out_path, text, self.permissions)
        return out_path

    async def _handle_handoff(
        self, last_step_output: AgentResult
    ) -> AgentResult | None:
        if self.loaded.spec.handoff.enabled and self.loaded.spec.handoff.agent_id:
            if isinstance(last_step_output, BaseModel):
                payload = last_step_output.model_dump_json(indent=2)
            elif isinstance(last_step_output, (dict, list)):
                payload = json.dumps(last_step_output, indent=2)
            else:
                payload = str(last_step_output)
            return await self._run_nested_agent(
                self.loaded.spec.handoff.agent_id, TextInput(payload)
            )
        return None


# ---------------------------------------------------------------------------
# Ollama-safe model subclass: patches content=None → content="" in assistant
# messages to work around Ollama rejecting null content as <nil>.
# See docs/tool_mode.md for detailed reasoning.
# ---------------------------------------------------------------------------
class OllamaSafeChatModel(OpenAIChatModel):
    """OpenAIChatModel that replaces content=None with content='' in assistant
    messages, preventing Ollama's 'invalid message content type: <nil>' error."""

    @dataclass
    class _MapModelResponseContext(OpenAIChatModel._MapModelResponseContext):
        def _into_message_param(self) -> chat.ChatCompletionAssistantMessageParam:
            message_param = super()._into_message_param()
            if message_param.get("content") is None:
                message_param["content"] = ""
            return message_param


def resolve_schema(loaded: LoadedAgentFile, schema_name: str) -> Type[BaseModel]:
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
