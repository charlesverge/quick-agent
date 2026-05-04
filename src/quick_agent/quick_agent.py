"""Agent execution logic."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Type, TypeAlias, TypedDict, cast
from uuid import uuid4

import httpx
import openai
from httpx._config import DEFAULT_LIMITS
from pydantic import BaseModel, JsonValue

from quick_agent.agent_config import AgentConfig
from quick_agent.agent_model_utils import resolve_schema
from quick_agent.agent_processor import AgentProcessor
from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.agent_utils import (
    agent_results_to_str,
    extract_finish_reason,
    normalize_usage_metrics,
    parse_structured_result,
)
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.executor import AgentExecutor
from quick_agent.input_adaptors import FileInput, InputAdaptor, TextInput
from quick_agent.io_utils import write_output as write_text
from quick_agent.json_utils import json_compatible_value, repair_json_text
from quick_agent.mapping.map_chunks import MapChunks
from quick_agent.mapping.map_paragraphs import MapParagraphs
from quick_agent.models.batch_request import (
    BatchAgentContext,
    BatchImportOutcome,
    BatchImportRequest,
    BatchMessage,
    BatchModelConfig,
    BatchSubmitRequest,
    BatchToolDefinition,
)
from quick_agent.models.chain_step_spec import ChainStepSpec, ToolChoice
from quick_agent.models.content_processing_spec import ChunkProcessingSpec
from quick_agent.models.loaded_agent_file import LoadedAgentFile
from quick_agent.models.model_spec import ModelSettings, ModelSpec
from quick_agent.models.run_input import RunInput
from quick_agent.output import write_output
from quick_agent.prompting import make_user_prompt
from quick_agent.recorder import Recorder
from quick_agent.samplers.simple_ratios import SampleRatios
from quick_agent.tools_loader import load_tool_definitions
from quick_agent.toolset import AgentToolset
from quick_agent.types import AgentResult, StepOutput

logger = logging.getLogger(__name__)


BatchCallHandler: TypeAlias = Callable[
    [BatchSubmitRequest], Awaitable[BatchImportRequest] | BatchImportRequest
]


class ChainState(TypedDict):
    agent_id: str
    steps: dict[str, StepOutput]
    last_step_output: StepOutput | None


class ToolRunDeps(TypedDict):
    state: ChainState
    memory: dict[str, Any]


class QuickAgentConfigError(ValueError):
    pass


class QuickAgent:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        tools: AgentTools,
        directory_permissions: DirectoryPermissions,
        agent_id: str,
        input_data: InputAdaptor | Path,
        extra_tools: list[str] | None = None,
        model: ModelSpec | None = None,
        write_output: bool = True,
        record_http_traffic: bool = False,
        enable_llm_request_logging: bool = False,
        llm_log_path: Path | str | None = None,
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, JsonValue] | None = None,
        memory: dict[str, Any] | None = None,
        client: openai.AsyncOpenAI | None = None,
        test_mode: bool = False,
    ) -> None:
        self._registry: AgentRegistry = registry
        self._tools: AgentTools = tools
        self._directory_permissions: DirectoryPermissions = directory_permissions
        self._input_data: InputAdaptor | Path = input_data
        self._extra_tools: list[str] | None = extra_tools
        self._param_extra_headers: dict[str, str] | None = extra_headers
        self.loaded: LoadedAgentFile = self._registry.get(agent_id)
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
        self.toolset: AgentToolset | None = self._build_toolset()
        self.model_spec: ModelSpec = model or self.loaded.spec.model
        self.extra_headers = self._merge_extra_headers()
        self._record_http_traffic: bool = record_http_traffic

        self._memory: dict[str, Any] = memory if memory is not None else {}
        logger.info(f"Initialized QuickAgent {agent_id}")
        self.state: ChainState = self._init_state(agent_id=agent_id)
        self._http_client = self._build_http_client()
        executor_config = AgentConfig(
            agent_id=agent_id,
            toolset=self.toolset,
            tool_ids=self.tool_ids,
            memory=self._memory,
            model_spec=self.model_spec,
            client=client,
            http_client=self._http_client,
            extra_headers=extra_headers,
            extra_body=extra_body,
            record_http_traffic=self._record_http_traffic,
            run_input=self.run_input,
            loaded=self.loaded,
            extra_tools=self._extra_tools,
            recorder=None,
            state=self.state,
        )
        self._executor = AgentExecutor(config=executor_config)
        self._recorder: Recorder = Recorder(
            executor=self._executor,
            enable_llm_request_logging=enable_llm_request_logging,
            llm_log_path=llm_log_path,
        )
        self._executor.config.recorder = self._recorder
        if self._record_http_traffic and self._executor.context.http_client is not None:
            self._executor.context.http_client.event_hooks = {
                "request": [self._recorder._record_http_request],
                "response": [self._recorder._record_http_response],
            }
        self.last_run_metrics: dict[str, object] | None = None
        self._test_mode: bool = test_mode

    @property
    def processor(self) -> AgentProcessor | None:
        if not self._test_mode:
            return None
        return AgentProcessor(self._executor)

    def _build_http_client(self) -> httpx.AsyncClient | None:
        timeout_seconds = self.model_spec.timeout_seconds or 60.0
        keepalive_expiry_seconds = self.model_spec.keepalive_expiry_seconds
        limits: httpx.Limits = DEFAULT_LIMITS
        if keepalive_expiry_seconds is not None:
            limits = httpx.Limits(
                max_connections=100, keepalive_expiry=keepalive_expiry_seconds
            )

        headers = self.extra_headers if self.extra_headers else None
        return httpx.AsyncClient(
            timeout=timeout_seconds,
            limits=limits,
            headers=headers,
        )

    def _merge_extra_headers(self) -> dict[str, str] | None:
        merged_headers = dict(self.model_spec.extra_headers or {})
        if self._param_extra_headers is not None:
            merged_headers.update(self._param_extra_headers)
        return merged_headers if merged_headers else None

    def load_batch_context(self, *, context: BatchAgentContext) -> None:
        state_obj = context.state
        agent_id_obj = state_obj.get("agent_id")
        steps_obj: dict[str, StepOutput] = cast(dict[str, StepOutput], state_obj.get("steps") or {})
        if not isinstance(agent_id_obj, str):
            raise ValueError("Invalid batch context state.")
        steps: dict[str, StepOutput] = {}
        for key, value in steps_obj.items():
            if not isinstance(key, str):
                raise ValueError(f"Invalid step key type in batch context: {type(key)}")
            if not isinstance(value, (str, dict)):
                raise ValueError(
                    f"Invalid step output type in batch context: {type(value)}"
                )
            steps[key] = value
        last_step_output: str | dict[str, object] | None
        last_step_output_obj = state_obj.get("last_step_output")
        if last_step_output_obj is None:
            last_step_output = None
        elif isinstance(last_step_output_obj, str):
            last_step_output = last_step_output_obj
        elif isinstance(last_step_output_obj, dict):
            last_step_output = last_step_output_obj
        else:
            raise ValueError(
                f"Invalid last_step_output type in batch context: {type(last_step_output_obj)}"
            )
        self.state = {
            "agent_id": agent_id_obj,
            "steps": steps,
            "last_step_output": last_step_output,
        }
        self._executor.config.state = self.state

    async def _write_output_handoff(self, output) -> AgentResult:
        if self._write_output_file:
            output_file = self.loaded.spec.output.file
            if output_file is None:
                raise ValueError("Output file is not configured.")
            write_output(output_file, output, self.permissions)
        handoff_output = await self._handle_handoff(output)
        if handoff_output is not None:
            return handoff_output
        return output

    async def run(self) -> AgentResult:
        self._executor.last_run_metrics = None
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
            return await self._write_output_handoff(chunk_output)
        if self._is_empty_agent_body():
            return await self._write_output_handoff(self.run_input.text)

        try:
            last_step_output = await self._run_chain()

            final_output: AgentResult = last_step_output
            if self.loaded.spec.output.return_compiled_output:
                final_output = self._compiled_output(last_step_output)
            final_output = self._finalize_output_contract(final_output)

            if self._write_output_file:
                output_file = self.loaded.spec.output.file
                if output_file is None:
                    raise ValueError("Output file is not configured.")
                write_output(output_file, final_output, self.permissions)

            handoff_output = await self._handle_handoff(last_step_output)
            if handoff_output is not None:
                return handoff_output

            return final_output
        finally:
            self._recorder._write_llm_request_log(None)

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
            write_text(Path(debug_output_file), sample_result, self.permissions)

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
            agent_id=self._executor.config.agent_id,
            input_data=TextInput(chunk_text),
            extra_tools=self._extra_tools,
            model=self.model_spec,
            write_output=False,
            record_http_traffic=self._record_http_traffic,
            enable_llm_request_logging=self._recorder._enable_llm_request_logging,
            llm_log_path=self._recorder._llm_log_path,
            extra_headers=self._executor.context.extra_headers,
            extra_body=self._executor.context.extra_body,
            client=self._executor.config.client,
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
            enable_llm_request_logging=self._recorder._enable_llm_request_logging,
            llm_log_path=self._recorder._llm_log_path,
            client=self._executor.config.client,
        )
        return await agent.run()

    def _init_state(self, *, agent_id: str) -> ChainState:
        return {
            "agent_id": agent_id,
            "steps": {},
            "last_step_output": None,
        }

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

    def _build_batch_messages(
        self,
        *,
        instructions: str | None,
        system_prompt: str | list[str],
        user_prompt: str,
    ) -> list[BatchMessage]:
        messages: list[BatchMessage] = []
        system_parts: list[str] = []
        if isinstance(system_prompt, str) and system_prompt:
            system_parts.append(system_prompt)
        elif isinstance(system_prompt, list):
            for item in system_prompt:
                if isinstance(item, str) and item:
                    system_parts.append(item)
        if isinstance(instructions, str) and instructions:
            system_parts.append(instructions)
        if system_parts:
            messages.append(
                BatchMessage(
                    role="system",
                    content="\n".join(system_parts),
                )
            )
        messages.append(BatchMessage(role="user", content=user_prompt))
        return messages

    def create_batch_request_for_current_step(
        self,
        *,
        step_id: str | None,
        step_kind: str,
        output_schema: str | None,
        instructions: str | None,
        system_prompt: str | list[str],
        user_prompt: str,
        model_settings: ModelSettings,
        max_tool_calls: int = 3,
    ) -> BatchSubmitRequest:
        configured_response_as_tool: bool | None = None
        if isinstance(model_settings.response_as_tool, bool):
            configured_response_as_tool = model_settings.response_as_tool
        response_as_tool = configured_response_as_tool is True
        response_format: dict[str, JsonValue] | None = None
        extra_body_obj = model_settings.extra_body
        if isinstance(extra_body_obj, dict):
            response_format_obj = extra_body_obj.get("response_format")
            if isinstance(response_format_obj, dict):
                response_format = response_format_obj
        if response_format is None and output_schema is not None:
            schema_cls = resolve_schema(self.loaded, output_schema)
            schema = schema_cls.model_json_schema()
            self._apply_strict_schema(schema)
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_cls.__name__,
                    "schema": schema,
                    "strict": True,
                },
            }
        request_id = f"{self._executor.config.agent_id}-{uuid4()}"
        state_obj = json_compatible_value(self.state)
        if not isinstance(state_obj, dict):
            raise ValueError("Expected chain state to be a JSON-compatible object.")
        state: dict[str, object] = {}
        for key, value in state_obj.items():
            state[str(key)] = value
        resolved_tool_choice = self._normalize_tool_choice(model_settings.tool_choice)
        tools = self._batch_tools()
        if resolved_tool_choice is not None:
            mode = resolved_tool_choice.mode
            if mode == "none":
                tools = []
            elif resolved_tool_choice.allowed_tools is not None:
                allowed_names = {ref.name for ref in resolved_tool_choice.allowed_tools}
                tools = [tool for tool in tools if tool.name in allowed_names]
        final_result_tool_enabled = False
        if response_format is not None and tools:
            if configured_response_as_tool is None and self._is_bedrock_request():
                response_as_tool = True
            if response_as_tool:
                response_schema = self._extract_response_schema(response_format)
                if any(tool.name == "final_result" for tool in tools):
                    raise ValueError(
                        "Configuration error: tool name 'final_result' is reserved when response_as_tool=true."
                    )
                tools.append(
                    BatchToolDefinition(
                        name="final_result",
                        description="Return the final structured response.",
                        input_schema=response_schema,
                        strict=True,
                    )
                )
                response_format = None
                final_result_tool_enabled = True
            elif self._is_bedrock_request():
                raise QuickAgentConfigError(
                    "Configuration error: Bedrock requests cannot use response_format and tools together. "
                    "Set response_as_tool=true at the agent or step level."
                )
        if self._is_bedrock_request():
            strict_tools: list[BatchToolDefinition] = []
            for tool in tools:
                tool_schema: dict[str, JsonValue] = {}
                for key, value in tool.input_schema.items():
                    tool_schema[str(key)] = value
                self._apply_strict_schema(tool_schema)
                strict_tools.append(
                    tool.model_copy(
                        update={"input_schema": tool_schema, "strict": True}
                    )
                )
            tools = strict_tools
        return BatchSubmitRequest(
            request_id=request_id,
            agent_id=self._executor.config.agent_id,
            step_id=step_id,
            step_kind=step_kind,
            output_schema=output_schema,
            model=BatchModelConfig(
                provider=self.model_spec.provider,
                base_url=self._executor.context.effective_base_url,
                model_name=self.model_spec.model_name,
                temperature=self.model_spec.temperature,
                max_completion_tokens=self.model_spec.max_completion_tokens,
                extra_headers=self._executor.context.extra_headers or None,
                extra_body=self._executor.context.extra_body or None,
                bedrock_request_mode=model_settings.bedrock_request_mode,
            ),
            messages=self._build_batch_messages(
                instructions=instructions,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ),
            response_format=response_format,
            tool_choice=resolved_tool_choice,
            max_tool_calls=max_tool_calls,
            tool_ids=list(self.tool_ids),
            tools=tools or None,
            tool_use_enabled=bool(tools),
            response_as_tool=response_as_tool,
            final_result_tool_enabled=final_result_tool_enabled,
            context=BatchAgentContext(
                input_text=self.run_input.text,
                state=state,
                memory=dict(self._memory),
                safe_dir=self.loaded.spec.safe_dir,
                extra_tools=list(self._extra_tools or []),
            ),
        )

    def batch(self) -> BatchSubmitRequest:
        self.model_spec.provider = "bedrock"
        if self.loaded.spec.chain:
            step_index = len(self.state["steps"])
            if step_index >= len(self.loaded.spec.chain):
                raise ValueError(
                    "No remaining chain steps for batch request generation."
                )
            step = self.loaded.spec.chain[step_index]
            step_prompt = self.loaded.step_prompts[step.prompt_section]
            step_instructions = self._build_step_instructions(step_prompt)
            model_settings = self._executor.context.model_settings_json
            if step.kind == "structured":
                if not step.output_schema:
                    raise ValueError(
                        f"Step {step.id} is structured but missing output_schema."
                    )
                schema_cls = resolve_schema(self.loaded, step.output_schema)
                model_settings = self._executor.context.build_structured_model_settings(
                    schema_cls=schema_cls
                )
            max_tool_calls = self._resolve_max_tool_calls(step)
            return self.create_batch_request_for_current_step(
                step_id=step.id,
                step_kind=step.kind,
                output_schema=step.output_schema,
                instructions=step_instructions,
                system_prompt=self.loaded.system_prompt,
                user_prompt=make_user_prompt(self.run_input, self.state),
                model_settings=self._resolve_model_settings(
                    model_settings=model_settings, step=step
                ),
                max_tool_calls=max_tool_calls,
            )

        single_schema = self.loaded.spec.output.output_schema
        model_settings = self._executor.context.model_settings_json
        if single_schema is not None:
            schema_cls = resolve_schema(self.loaded, single_schema)
            model_settings = self._executor.context.build_structured_model_settings(
                schema_cls=schema_cls
            )
        return self.create_batch_request_for_current_step(
            step_id=None,
            step_kind="single_shot",
            output_schema=single_schema,
            instructions=self.loaded.instructions,
            system_prompt=self.loaded.system_prompt,
            user_prompt=self._build_single_shot_prompt(),
            model_settings=self._resolve_model_settings(
                model_settings=model_settings, step=None
            ),
            max_tool_calls=self._resolve_max_tool_calls(None),
        )

    async def import_result(
        self, *, batch_import: BatchImportRequest
    ) -> BatchImportOutcome:
        self.model_spec.provider = "bedrock"
        outcome = self._executor.import_outcome(batch_import=batch_import)
        if outcome.tool_calls is not None:
            pending = outcome.pending_submit_request
            if pending is None:
                raise ValueError("tool_use outcome is missing pending_submit_request.")
            if pending.tool_call_rounds() >= pending.max_tool_calls:
                raise ValueError(
                    f"Max tool call rounds reached for request_id={pending.request_id}: "
                    f"max_tool_calls={pending.max_tool_calls}"
                )
            executed = await self._executor._execute_tool_calls(outcome.tool_calls)
            next_request = self._executor._build_next_request_with_tool_results(
                tool_calls=outcome.tool_calls,
                executed=executed,
                submit_request=pending,
            )
            return BatchImportOutcome(next_request=next_request)
        if outcome.next_request is not None:
            return outcome
        raw_result = outcome.result
        if raw_result is None:
            raise ValueError("Batch import outcome did not include a result.")
        result_outcome = (
            self._import_chain_result(raw_result)
            if self.loaded.spec.chain
            else self._import_single_shot_result(raw_result)
        )
        result = result_outcome.result
        if result is None:
            raise ValueError("Import result did not produce output.")
        if result_outcome.next_request is not None:
            return BatchImportOutcome(
                result=result, next_request=result_outcome.next_request
            )
        finalized = self._finalize_output_contract(result)
        if self._write_output_file:
            output_file = self.loaded.spec.output.file
            if output_file is None:
                raise ValueError("Output file is not configured.")
            write_output(output_file, finalized, self.permissions)
        return BatchImportOutcome(result=finalized)

    def _import_single_shot_result(self, raw_result: AgentResult) -> BatchImportOutcome:
        schema_name = self.loaded.spec.output.output_schema
        step_kind = "text" if schema_name is None else "structured"
        parsed = self._parse_import_result(
            raw_result=raw_result,
            step_kind=step_kind,
            output_schema=schema_name,
            step_id=None,
        )
        return BatchImportOutcome(result=parsed)

    def _import_chain_result(self, raw_result: AgentResult) -> BatchImportOutcome:
        chain = self.loaded.spec.chain
        step_index = len(self.state["steps"])
        if step_index >= len(chain):
            raise ValueError("No remaining chain steps for imported batch result.")
        step = chain[step_index]
        parsed_obj = self._parse_import_result(
            raw_result=raw_result,
            step_kind=step.kind,
            output_schema=step.output_schema,
            step_id=step.id,
        )
        parsed: BaseModel | str
        if step.kind == "structured":
            if not isinstance(parsed_obj, BaseModel):
                raise ValueError(
                    f"Structured step {step.id} did not return a BaseModel output."
                )
            parsed = parsed_obj
        else:
            if not isinstance(parsed_obj, str):
                raise ValueError(f"Text step {step.id} did not return a string output.")
            parsed = parsed_obj
        self.state["steps"][step.id] = parsed
        self.state["last_step_output"] = parsed
        next_index = step_index + 1
        if next_index < len(chain):
            next_step = chain[next_index]
            next_prompt = self.loaded.step_prompts[next_step.prompt_section]
            next_instructions = self._build_step_instructions(next_prompt)
            next_model_settings = self._executor.context.model_settings_json
            if next_step.kind == "structured":
                if not next_step.output_schema:
                    raise ValueError(
                        f"Step {next_step.id} is structured but missing output_schema."
                    )
                next_schema_cls = resolve_schema(self.loaded, next_step.output_schema)
                next_model_settings = (
                    self._executor.context.build_structured_model_settings(
                        schema_cls=next_schema_cls
                    )
                )
            next_request = self.create_batch_request_for_current_step(
                step_id=next_step.id,
                step_kind=next_step.kind,
                output_schema=next_step.output_schema,
                instructions=next_instructions,
                system_prompt=self.loaded.system_prompt,
                user_prompt=make_user_prompt(self.run_input, self.state),
                model_settings=self._resolve_model_settings(
                    model_settings=next_model_settings, step=next_step
                ),
                max_tool_calls=self._resolve_max_tool_calls(next_step),
            )
            return BatchImportOutcome(result=parsed, next_request=next_request)
        final_result: AgentResult
        if self.loaded.spec.output.return_compiled_output:
            final_result = self._compiled_output(parsed)
        else:
            final_result = parsed
        return BatchImportOutcome(result=final_result)

    def _parse_import_result(
        self,
        *,
        raw_result: AgentResult,
        step_kind: str,
        output_schema: str | None,
        step_id: str | None,
    ) -> AgentResult:
        if step_kind == "text":
            if not isinstance(raw_result, str):
                if step_id is None and isinstance(raw_result, dict):
                    return raw_result
                if step_id is None:
                    raise ValueError("Single-shot text output must be a string.")
                raise ValueError(f"Text step {step_id} expected a string output.")
            return raw_result
        if step_kind == "structured":
            if not output_schema:
                if step_id is None:
                    raise ValueError("Structured output is missing output_schema.")
                raise ValueError(
                    f"Step {step_id} is structured but missing output_schema."
                )
            if not isinstance(raw_result, (BaseModel, str, dict)):
                if step_id is None:
                    raise ValueError(
                        "Structured output expected BaseModel, JSON string, or object output."
                    )
                raise ValueError(
                    f"Structured step {step_id} expected BaseModel, JSON string, or object output."
                )
            schema_cls = resolve_schema(self.loaded, output_schema)
            return parse_structured_result(raw_result, schema_cls)
        raise ValueError(f"Unsupported import step kind: {step_kind}")

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

    def _finalize_output_contract(self, output: AgentResult) -> AgentResult:
        output_schema = self.loaded.spec.output.output_schema
        output_format = self.loaded.spec.output.format
        if output_schema is not None:
            schema_cls = resolve_schema(self.loaded, output_schema)
            if isinstance(output, BaseModel):
                if isinstance(output, schema_cls):
                    return output
                payload = output.model_dump(mode="json")
                return schema_cls.model_validate(payload)
            if isinstance(output, (str, dict)):
                return parse_structured_result(output, schema_cls)
            raise ValueError("Structured output requires schema-compatible result.")
        if output_format == "json":
            if isinstance(output, BaseModel):
                return output
            if isinstance(output, dict):
                return output
            if isinstance(output, str):
                try:
                    parsed = json.loads(output)
                except json.JSONDecodeError:
                    print(output)
                    cleaned_raw = repair_json_text(output)
                    parsed = json.loads(cleaned_raw)
                if not isinstance(parsed, dict):
                    raise ValueError("JSON output must be a JSON object.")
                return parsed
            raise ValueError("JSON output requires a JSON object result.")
        if output_format == "markdown":
            if not isinstance(output, str):
                raise ValueError(
                    f"Text output must be a string (format=markdown, got {type(output).__name__})."
                )
            return output
        if output_format == "structured":
            if not isinstance(output, BaseModel):
                raise ValueError("Structured output requires a BaseModel result.")
            return output
        if not isinstance(output, str):
            raise ValueError(
                f"Text output must be a string (format=text, got {type(output).__name__})."
            )
        return output

    def _capture_metrics(self, *, usage: object, response: object | None) -> None:
        model = self.model_spec.model_name
        if response is not None:
            response_model = getattr(response, "model", None)
            if isinstance(response_model, str) and response_model:
                model = response_model
        metrics: dict[str, object] = {
            "provider": self.model_spec.provider,
            "model": model,
            "usage": normalize_usage_metrics(usage),
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
            finish_reason = extract_finish_reason(response)
            if finish_reason is not None:
                metrics["finish_reason"] = finish_reason
        self._executor.last_run_metrics = metrics

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

    def _unexpected_model_behavior_request_context(
        self,
        *,
        instructions: str | None,
        system_prompt: str | list[str],
        user_prompt: str,
        model_settings: ModelSettings,
    ) -> dict[str, object]:
        context: dict[str, object] = {
            "base_url": self._executor.context.effective_base_url,
            "model_name": self.model_spec.model_name,
            "instructions": instructions,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model_settings": json_compatible_value(model_settings),
        }
        context.update(self._recorder._last_http_exchange_context())
        return context

    async def _run_text_step(
        self,
        *,
        step: ChainStepSpec,
    ) -> str:
        prefix = "QuickAgent._run_text_step"
        user_prompt = make_user_prompt(self.run_input, self.state)
        step_prompt = self.loaded.step_prompts[step.prompt_section]
        step_instructions = self._build_step_instructions(step_prompt)
        self._recorder._record_llm_request(
            call_site="run_text_step",
            step_id=step.id,
            step_kind=step.kind,
            output_schema=step.output_schema,
            instructions=step_instructions,
            system_prompt=self.loaded.system_prompt,
            user_prompt=user_prompt,
            model_settings=self._executor.context.model_settings_json,
        )
        logger.info(
            "%s: model=%s step=%s > Calling model",
            prefix,
            self.model_spec.model_name,
            step.id,
        )
        batch_request = self.create_batch_request_for_current_step(
            step_id=step.id,
            step_kind=step.kind,
            output_schema=step.output_schema,
            instructions=step_instructions,
            system_prompt=self.loaded.system_prompt,
            user_prompt=user_prompt,
            model_settings=self._resolve_model_settings(
                model_settings=self._executor.context.model_settings_json, step=step
            ),
            max_tool_calls=self._resolve_max_tool_calls(step),
        )
        output = await self._executor._execute_batch_request(
            batch_request=batch_request,
            schema_cls=None,
        )
        if not isinstance(output, str):
            raise ValueError("Text step expected a string output.")
        return output

    async def _run_single_shot(self) -> BaseModel | str:
        prefix = "QuickAgent._run_single_shot"
        schema_name = self.loaded.spec.output.output_schema
        schema_cls: Type[BaseModel] | None = None
        if schema_name:
            schema_cls = resolve_schema(self.loaded, schema_name)
        logger.info("%s: model=%s > Calling model", prefix, self.model_spec.model_name)
        user_prompt = self._build_single_shot_prompt()
        instructions = self.loaded.instructions
        system_prompt = self.loaded.system_prompt
        model_settings = self._executor.context.model_settings_json
        if schema_cls is not None:
            model_settings = self._executor.context.build_structured_model_settings(
                schema_cls=schema_cls
            )
        self._recorder._record_llm_request(
            call_site="run_single_shot",
            step_id=None,
            step_kind="single_shot",
            output_schema=self.loaded.spec.output.output_schema,
            instructions=instructions,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_settings=model_settings,
        )
        batch_request = self.create_batch_request_for_current_step(
            step_id=None,
            step_kind="single_shot",
            output_schema=self.loaded.spec.output.output_schema,
            instructions=instructions,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_settings=self._resolve_model_settings(
                model_settings=model_settings, step=None
            ),
            max_tool_calls=self._resolve_max_tool_calls(None),
        )
        return await self._executor._execute_batch_request(
            batch_request=batch_request,
            schema_cls=schema_cls,
        )

    async def _run_structured_step(
        self,
        *,
        step: ChainStepSpec,
    ) -> BaseModel:
        prefix = "QuickAgent._run_structured_step"
        if not step.output_schema:
            raise ValueError(f"Step {step.id} is structured but missing output_schema.")
        schema_cls = resolve_schema(self.loaded, step.output_schema)

        model_settings = self._executor.context.build_structured_model_settings(
            schema_cls=schema_cls
        )

        user_prompt = make_user_prompt(self.run_input, self.state)
        step_prompt = self.loaded.step_prompts[step.prompt_section]
        step_instructions = self._build_step_instructions(step_prompt)
        self._recorder._record_llm_request(
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
        batch_request = self.create_batch_request_for_current_step(
            step_id=step.id,
            step_kind=step.kind,
            output_schema=step.output_schema,
            instructions=step_instructions,
            system_prompt=self.loaded.system_prompt,
            user_prompt=user_prompt,
            model_settings=self._resolve_model_settings(
                model_settings=model_settings, step=step
            ),
            max_tool_calls=self._resolve_max_tool_calls(step),
        )
        output = await self._executor._execute_batch_request(
            batch_request=batch_request,
            schema_cls=schema_cls,
        )
        if not isinstance(output, BaseModel):
            raise ValueError("Structured step expected a structured output.")
        return output

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

    def _build_toolset(self) -> AgentToolset | None:
        if not self.has_tools():
            return None
        return self._tools.build_toolset(self.tool_ids, self.permissions)

    def _batch_tools(self) -> list[BatchToolDefinition]:
        tool_ids = [t for t in self.tool_ids if t != "agent_call"]
        if not tool_ids or not self._tools._tool_roots:
            return []
        tools = load_tool_definitions(self._tools._tool_roots, tool_ids)
        normalized: list[BatchToolDefinition] = []
        for tool in tools:
            normalized.append(tool.model_copy(update={"strict": False}))
        return normalized

    def _resolve_tool_choice(self, step: ChainStepSpec | None) -> ToolChoice | None:
        if step is not None and step.tool_choice is not None:
            return step.tool_choice
        return self.loaded.spec.tool_choice

    def _resolve_response_as_tool(self, step: ChainStepSpec | None) -> bool | None:
        if step is not None and step.response_as_tool is not None:
            return step.response_as_tool
        return self.loaded.spec.response_as_tool

    def _with_response_as_tool(
        self,
        *,
        model_settings: ModelSettings,
        step: ChainStepSpec | None,
    ) -> ModelSettings:
        response_as_tool = self._resolve_response_as_tool(step)
        return model_settings.model_copy(update={"response_as_tool": response_as_tool})

    def _with_tool_choice(
        self,
        *,
        model_settings: ModelSettings,
        step: ChainStepSpec | None,
    ) -> ModelSettings:
        tool_choice = self._resolve_tool_choice(step)
        return model_settings.model_copy(update={"tool_choice": tool_choice})

    def _resolve_model_settings(
        self,
        *,
        model_settings: ModelSettings,
        step: ChainStepSpec | None,
    ) -> ModelSettings:
        return self._with_tool_choice(
            model_settings=self._with_response_as_tool(
                model_settings=model_settings, step=step
            ),
            step=step,
        )

    def _extract_response_schema(
        self, response_format: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        response_type = response_format.get("type")
        if response_type != "json_schema":
            raise ValueError(
                "Configuration error: response_as_tool requires response_format.type='json_schema'."
            )
        json_schema_obj = response_format.get("json_schema")
        if not isinstance(json_schema_obj, dict):
            raise ValueError(
                "Configuration error: response_as_tool requires response_format.json_schema."
            )
        schema_obj = json_schema_obj.get("schema")
        if not isinstance(schema_obj, dict):
            raise ValueError(
                "Configuration error: response_as_tool requires response_format.json_schema.schema."
            )
        schema: dict[str, JsonValue] = {}
        for key, value in schema_obj.items():
            schema[str(key)] = value
        return schema

    def _is_bedrock_request(self) -> bool:
        return self.model_spec.provider == "bedrock"

    def _normalize_tool_choice(
        self, tool_choice: ToolChoice | None
    ) -> ToolChoice | None:
        if tool_choice is None:
            return None
        if tool_choice.mode == "any" and self.model_spec.provider != "bedrock":
            return tool_choice.model_copy(update={"mode": "auto"})
        return tool_choice

    def _resolve_max_tool_calls(self, step: ChainStepSpec | None) -> int:
        if step is not None and step.max_tool_calls is not None:
            return step.max_tool_calls
        if self.loaded.spec.max_tool_calls is not None:
            return self.loaded.spec.max_tool_calls
        return 3

    def _toolsets_for_run(self) -> list[AgentToolset]:
        if not self.has_tools():
            return []
        if self.toolset is None:
            return []
        return [self.toolset]

    def _tool_deps(self) -> ToolRunDeps:
        return {"state": self.state, "memory": self._executor.config.memory}

    async def _handle_handoff(
        self, last_step_output: AgentResult
    ) -> AgentResult | None:
        if self.loaded.spec.handoff.enabled and self.loaded.spec.handoff.agent_id:
            payload = agent_results_to_str(last_step_output)
            return await self._run_nested_agent(
                self.loaded.spec.handoff.agent_id, TextInput(payload)
            )
        return None
