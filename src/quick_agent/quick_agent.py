"""Agent execution logic."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Type, TypeAlias, TypedDict

logger = logging.getLogger(__name__)

import httpx
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.exceptions import QuickAgentChatNotSupportedException
from quick_agent.exceptions import QuickAgentException
from quick_agent.exceptions import QuickAgentToolsNotSupportedException
from quick_agent.exceptions import QuickAgentUnexpectedModelBehaviorException
from quick_agent.input_adaptors import FileInput, InputAdaptor, TextInput
from quick_agent.io_utils import write_output
from quick_agent.json_utils import extract_first_json_object
from quick_agent.models.loaded_agent_file import LoadedAgentFile
from quick_agent.models.chain_step_spec import ChainStepSpec
from quick_agent.models.model_spec import ModelSpec
from quick_agent.models.run_input import RunInput
from quick_agent.prompting import make_user_prompt
from quick_agent.single_shot import run_single_shot
from quick_agent.tools_loader import import_symbol

StepOutput: TypeAlias = str | dict[str, Any]


class ChainState(TypedDict):
    agent_id: str
    steps: dict[str, StepOutput]
    final_output: StepOutput | None


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
        self.permissions: DirectoryPermissions = self._directory_permissions.scoped(safe_dir)
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
        self._http_client: httpx.AsyncClient | None = self._build_http_client()
        self.model: OpenAIChatModel = build_model(self.model_spec, http_client=self._http_client)
        self.model_settings_json: ModelSettings | None = self._build_model_settings(self.model_spec)
        self.state: ChainState = self._init_state()
        self._enable_llm_request_logging: bool = enable_llm_request_logging
        if llm_log_path is None:
            llm_log_path = Path("log/results.log")
        self._llm_log_path: Path = Path(llm_log_path)

    async def run(self) -> BaseModel | str:
        if self.has_tools():
            if self.toolset is None:
                raise ValueError("Toolset is missing while tools are enabled.")
            self._tools.maybe_inject_agent_call(
                self.tool_ids,
                self.toolset,
                self.run_input.source_path,
                self._run_nested_agent,
            )

        try:
            final_output = await self._run_chain()

            if self._write_output_file:
                self._write_final_output(final_output)

            await self._handle_handoff(final_output)

            return final_output
        finally:
            self._write_llm_request_log(None)

    async def _run_nested_agent(self, agent_id: str, input_data: InputAdaptor | Path) -> BaseModel | str:
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
        )
        return await agent.run()

    def _init_state(self) -> ChainState:
        return {
            "agent_id": self._agent_id,
            "steps": {},
            "final_output": None,
        }

    def _build_model_settings(self, model_spec: ModelSpec) -> ModelSettings | None:
        if model_spec.provider == "openai-compatible":
            # Ollama OpenAI-compatible API uses "format": "json" to force JSON output.
            if model_spec.base_url != "https://api.openai.com/v1":
                return {"extra_body": {"format": "json"}}
        return None

    def _build_http_client(self) -> httpx.AsyncClient | None:
        timeout_seconds = self.model_spec.timeout_seconds or 60.0
        keepalive_expiry_seconds = self.model_spec.keepalive_expiry_seconds or 60.0
        limits: httpx.Limits | None = None
        if keepalive_expiry_seconds is not None:
            limits = httpx.Limits(keepalive_expiry=keepalive_expiry_seconds)
        if self._record_http_traffic:
            if timeout_seconds is not None and limits is not None:
                return httpx.AsyncClient(
                    event_hooks={
                        "request": [self._record_http_request],
                        "response": [self._record_http_response],
                    },
                    timeout=timeout_seconds,
                    limits=limits,
                )
            if timeout_seconds is not None:
                return httpx.AsyncClient(
                    event_hooks={
                        "request": [self._record_http_request],
                        "response": [self._record_http_response],
                    },
                    timeout=timeout_seconds,
                )
            if limits is not None:
                return httpx.AsyncClient(
                    event_hooks={
                        "request": [self._record_http_request],
                        "response": [self._record_http_response],
                    },
                    limits=limits,
                )
            return httpx.AsyncClient(
                event_hooks={
                    "request": [self._record_http_request],
                    "response": [self._record_http_response],
                }
            )
        if timeout_seconds is not None and limits is not None:
            return httpx.AsyncClient(timeout=timeout_seconds, limits=limits)
        if timeout_seconds is not None:
            return httpx.AsyncClient(timeout=timeout_seconds)
        if limits is not None:
            return httpx.AsyncClient(limits=limits)
        return None

    def _build_structured_model_settings(self, *, schema_cls: Type[BaseModel]) -> ModelSettings | None:
        model_settings: ModelSettings | None = self.model_settings_json
        provider = getattr(self.model, "provider", None)
        base_url = getattr(provider, "base_url", None)
        if base_url == "https://api.openai.com/v1":
            if self.model_settings_json is None:
                model_settings_dict: ModelSettings = {}
            else:
                model_settings_dict = self.model_settings_json
            extra_body_obj = model_settings_dict.get("extra_body")
            extra_body: dict[str, Any] = {}
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
    ) -> tuple[StepOutput, BaseModel | str]:
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

    def _record_llm_request(
        self,
        *,
        step_id: str | None,
        step_kind: str,
        output_schema: str | None,
        instructions: str | None,
        system_prompt: str | list[str],
        user_prompt: str,
        model_settings: ModelSettings | None,
    ) -> None:
        payload: dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_state": "before_request_start",
            "agent_id": self._agent_id,
            "model": {
                "provider": self.model_spec.provider,
                "base_url": self.model_spec.base_url,
                "model_name": self.model_spec.model_name,
            },
            "step": {
                "id": step_id,
                "kind": step_kind,
                "output_schema": output_schema,
            },
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
            logger.exception("%s: file=%s > Failed to write LLM request log", prefix, self._llm_log_path)

    def _normalize_agent_text(self, text: str) -> str | None:
        if text:
            return text
        return None

    def _current_model_settings(self) -> ModelSettings | None:
        return self.model_settings_json

    def _normalize_system_prompt(self, text: str) -> str | list[str]:
        if text:
            return text
        return []

    def _map_model_http_error(self, error: ModelHTTPError) -> QuickAgentException | None:
        body = error.body
        message = ""
        if isinstance(body, dict):
            body_message = body.get("message")
            if isinstance(body_message, str):
                message = body_message
        elif isinstance(body, str):
            message = body
        if "does not support tools" in message:
            return QuickAgentToolsNotSupportedException(model_name=error.model_name, message=message)
        if "does not support chat" in message:
            return QuickAgentChatNotSupportedException(model_name=error.model_name, message=message)
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
                    exchange_context: dict[str, object] = {"request": request_obj, "request_source": "quick_agent_http_traffic_log"}
                    if isinstance(response_obj, dict):
                        exchange_context["response"] = response_obj
                    return exchange_context
        for entry in reversed(self._http_traffic_entries):
            if entry.get("event") == "request":
                request_obj = entry.get("request")
                if isinstance(request_obj, dict):
                    return {"request": request_obj, "request_source": "quick_agent_http_traffic_log"}
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
            "base_url": self.model_spec.base_url,
            "model_name": self.model_spec.model_name,
            "instructions": instructions,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model_settings": self._json_compatible_value(model_settings),
        }
        context.update(self._last_http_exchange_context())
        return context

    async def _run_text_step(
        self,
        *,
        step: ChainStepSpec,
    ) -> tuple[StepOutput, BaseModel | str]:
        prefix = "QuickAgent._run_text_step"
        user_prompt = make_user_prompt(self.run_input, self.state)
        step_prompt = self.loaded.step_prompts[step.prompt_section]
        step_instructions = self._build_step_instructions(step_prompt)
        toolsets = self._toolsets_for_run()
        agent = Agent(
            self.model,
            instructions=step_instructions,
            system_prompt=self._normalize_system_prompt(self.loaded.system_prompt),
            toolsets=toolsets,
            output_type=str,
        )
        self._record_llm_request(
            step_id=step.id,
            step_kind=step.kind,
            output_schema=step.output_schema,
            instructions=step_instructions,
            system_prompt=self._normalize_system_prompt(self.loaded.system_prompt),
            user_prompt=user_prompt,
            model_settings=self._current_model_settings(),
        )
        logger.info("%s: model=%s step=%s > Calling model", prefix, self.model_spec.model_name, step.id)
        try:
            result = await agent.run(user_prompt)
        except UnexpectedModelBehavior as error:
            raise QuickAgentUnexpectedModelBehaviorException(
                original_exception=error,
                request_context=self._unexpected_model_behavior_request_context(
                    instructions=step_instructions,
                    system_prompt=self._normalize_system_prompt(self.loaded.system_prompt),
                    user_prompt=user_prompt,
                    model_settings=self._current_model_settings(),
                ),
            ) from error
        except ModelHTTPError as error:
            mapped_error = self._map_model_http_error(error)
            if mapped_error is not None:
                raise mapped_error from error
            raise error
        return result.output, result.output

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
    ) -> tuple[StepOutput, BaseModel | str]:
        prefix = "QuickAgent._run_structured_step"
        if not step.output_schema:
            raise ValueError(f"Step {step.id} is structured but missing output_schema.")
        schema_cls = resolve_schema(self.loaded, step.output_schema)

        model_settings = self._build_structured_model_settings(schema_cls=schema_cls)

        user_prompt = make_user_prompt(self.run_input, self.state)
        step_prompt = self.loaded.step_prompts[step.prompt_section]
        step_instructions = self._build_step_instructions(step_prompt)
        toolsets = self._toolsets_for_run()
        agent = Agent(
            self.model,
            instructions=step_instructions,
            system_prompt=self._normalize_system_prompt(self.loaded.system_prompt),
            toolsets=toolsets,
            output_type=schema_cls,
            model_settings=model_settings,
        )
        self._record_llm_request(
            step_id=step.id,
            step_kind=step.kind,
            output_schema=step.output_schema,
            instructions=step_instructions,
            system_prompt=self._normalize_system_prompt(self.loaded.system_prompt),
            user_prompt=user_prompt,
            model_settings=model_settings,
        )
        logger.info("%s: model=%s step=%s schema=%s > Calling model", prefix, self.model_spec.model_name, step.id, step.output_schema)
        try:
            result = await agent.run(user_prompt)
        except UnexpectedModelBehavior as error:
            raise QuickAgentUnexpectedModelBehaviorException(
                original_exception=error,
                request_context=self._unexpected_model_behavior_request_context(
                    instructions=step_instructions,
                    system_prompt=self._normalize_system_prompt(self.loaded.system_prompt),
                    user_prompt=user_prompt,
                    model_settings=model_settings,
                ),
            ) from error
        except ModelHTTPError as error:
            mapped_error = self._map_model_http_error(error)
            if mapped_error is not None:
                raise mapped_error from error
            raise error
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
        return parsed.model_dump(), parsed

    async def _run_chain(
        self,
    ) -> BaseModel | str:
        if not self.loaded.spec.chain:
            return await self._run_single_shot()
        final_output: BaseModel | str = ""
        for step in self.loaded.spec.chain:
            step_out, step_final = await self._run_step(
                step=step,
            )
            self.state["steps"][step.id] = step_out
            self.state["final_output"] = step_out
            final_output = step_final
        return final_output

    def has_tools(self) -> bool:
        if not self.tool_ids:
            return False
        return True

    def _build_tool_ids(self) -> list[str]:
        if not self.loaded.spec.tools:
            return []
        return list(dict.fromkeys((self.loaded.spec.tools or []) + (self._extra_tools or [])))

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

    def _write_final_output(self, final_output: BaseModel | str) -> Path:
        output_file = self.loaded.spec.output.file
        if not output_file:
            raise ValueError("Output file is not configured.")
        out_path = Path(output_file)
        if isinstance(final_output, BaseModel):
            if self.loaded.spec.output.format == "json":
                write_output(out_path, final_output.model_dump_json(indent=2), self.permissions)
            else:
                write_output(out_path, final_output.model_dump_json(indent=2), self.permissions)
        else:
            write_output(out_path, str(final_output), self.permissions)
        return out_path

    async def _handle_handoff(self, final_output: BaseModel | str) -> None:
        if self.loaded.spec.handoff.enabled and self.loaded.spec.handoff.agent_id:
            if isinstance(final_output, BaseModel):
                payload = final_output.model_dump_json(indent=2)
            else:
                payload = str(final_output)
            await self._run_nested_agent(self.loaded.spec.handoff.agent_id, TextInput(payload))


def resolve_schema(loaded: LoadedAgentFile, schema_name: str) -> Type[BaseModel]:
    if schema_name not in loaded.spec.schemas:
        raise KeyError(f"Schema {schema_name!r} not registered in agent.md schemas.")
    cls = import_symbol(loaded.spec.schemas[schema_name])
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        raise TypeError(f"Schema {schema_name!r} must be a Pydantic BaseModel subclass.")
    return cls


def build_model(model_spec: ModelSpec, *, http_client: httpx.AsyncClient | None = None) -> OpenAIChatModel:
    api_key = os.environ.get(model_spec.api_key_env, "noop")
    if http_client is None:
        provider = OpenAIProvider(base_url=model_spec.base_url, api_key=api_key)
    else:
        provider = OpenAIProvider(base_url=model_spec.base_url, api_key=api_key, http_client=http_client)
    return OpenAIChatModel(model_spec.model_name, provider=provider)
