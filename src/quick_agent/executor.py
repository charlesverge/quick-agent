from __future__ import annotations

import inspect
import json
import logging
import os
import typing
from dataclasses import dataclass
from typing import Any, Callable, Type
from uuid import uuid4

import openai
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params.function_definition import FunctionDefinition
from pydantic import BaseModel, JsonValue

from quick_agent.agent_config import AgentConfig
from quick_agent.agent_execution_context import AgentExecutionContext
from quick_agent.agent_state import AgentState
from quick_agent.agent_tool_schema import (
    strip_agent_state_from_schema,
    takes_agent_state,
)

from .agent_utils import (
    _as_agent_result,
    extract_finish_reason,
    normalize_tool_calls,
    normalize_usage_metrics,
    parse_structured_result,
)
from .exceptions import (
    QuickAgentChatNotSupportedException,
    QuickAgentException,
    QuickAgentToolsNotSupportedException,
)
from .json_utils import json_compatible_value
from .models.batch_request import (
    BatchAgentContext,
    BatchImportOutcome,
    BatchImportRequest,
    BatchMessage,
    BatchSubmitRequest,
)

logger = logging.getLogger(__name__)


def _should_convert_null(base_url: str) -> bool:
    """Determine if null content should be converted to empty string.

    Ollama rejects messages with null content, converting to "" prevents errors.
    Default behavior: convert for Ollama endpoints, not for OpenAI.
    """
    return "ollama" in base_url.lower()


@dataclass
class ToolCallResult:
    id: str
    name: str
    content: str | None = None
    error: str | None = None


class AgentExecutor:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.context = AgentExecutionContext.from_config(config)
        self.last_run_metrics: dict[str, object] | None = None

    def _capture_metrics(self, *, usage: object, response: object | None) -> None:
        model = self.config.model_spec.model_name
        if response is not None:
            response_model = getattr(response, "model", None)
            if isinstance(response_model, str) and response_model:
                model = response_model
        metrics: dict[str, object] = {
            "provider": self.config.model_spec.provider,
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
        self.last_run_metrics = metrics

    async def _execute_tool_calls(
        self, tool_calls: list[dict[str, object]]
    ) -> list[ToolCallResult]:
        prefix = "AgentExecutor._execute_tool_calls"
        results: list[ToolCallResult] = []
        toolset = self.config.toolset
        for tc in tool_calls:
            tc_id = tc.get("id")
            tc_name = tc.get("name")
            tc_args = tc.get("arguments")
            if not isinstance(tc_id, str) or not isinstance(tc_name, str):
                raise ValueError(f"Invalid tool call structure: {tc}")
            if toolset is None:
                results.append(
                    ToolCallResult(
                        id=tc_id, name=tc_name, error="Toolset not available."
                    )
                )
                continue
            tool = toolset.tools.get(tc_name)
            if tool is None:
                logger.warning(
                    f"{prefix}: tool_id={tc_id} name={tc_name} > Tool not found"
                )
                results.append(
                    ToolCallResult(
                        id=tc_id,
                        name=tc_name,
                        error=f"Tool '{tc_name}' not found.",
                    )
                )
                continue
            args: dict[str, object] = {}
            if isinstance(tc_args, dict):
                args = tc_args
            elif isinstance(tc_args, str):
                try:
                    parsed = json.loads(tc_args)
                    if isinstance(parsed, dict):
                        args = parsed
                except (json.JSONDecodeError, ValueError):
                    pass
            try:
                plain_func: Callable[..., Any] = tool.function
                if takes_agent_state(plain_func):
                    state = AgentState(memory=self.config.memory)
                    if inspect.iscoroutinefunction(plain_func):
                        output = await plain_func(state, **args)
                    else:
                        output = plain_func(state, **args)
                elif inspect.iscoroutinefunction(plain_func):
                    output = await plain_func(**args)
                else:
                    output = plain_func(**args)
                if isinstance(output, dict):
                    text_val = output.get("text")
                    content = (
                        str(text_val) if text_val is not None else json.dumps(output)
                    )
                else:
                    content = str(output)
                logger.info(
                    f"{prefix}: tool_id={tc_id} name={tc_name} > result_length={len(content)}"
                )
                results.append(ToolCallResult(id=tc_id, name=tc_name, content=content))
            except Exception as exc:
                logger.error(f"{prefix}: tool_id={tc_id} name={tc_name} > error={exc}")
                results.append(ToolCallResult(id=tc_id, name=tc_name, error=str(exc)))
        return results

    def _build_next_request_with_tool_results(
        self,
        *,
        tool_calls: list[dict[str, object]],
        executed: list[ToolCallResult],
        submit_request: BatchSubmitRequest,
    ) -> BatchSubmitRequest:
        messages: list[BatchMessage] = list(submit_request.messages)
        oa_tool_calls: list[dict[str, JsonValue]] = []

        for tc in tool_calls:
            tc_id_obj = tc.get("id")
            tc_name_obj = tc.get("name")
            tc_args = tc.get("arguments")
            tc_id_str: str | None = tc_id_obj if isinstance(tc_id_obj, str) else None
            tc_name_str: str | None = (
                tc_name_obj if isinstance(tc_name_obj, str) else None
            )
            args_str: str
            if isinstance(tc_args, dict):
                args_str = json.dumps(tc_args)
            elif isinstance(tc_args, str):
                args_str = tc_args
            else:
                args_str = "{}"
            oa_tool_calls.append(
                {
                    "id": tc_id_str,
                    "type": "function",
                    "function": {
                        "name": tc_name_str,
                        "arguments": args_str,
                    },
                }
            )
        messages.append(BatchMessage(role="assistant", tool_calls=oa_tool_calls))

        for tc, result in zip(tool_calls, executed):
            tc_name_obj = tc.get("name")
            tc_name: str | None = tc_name_obj if isinstance(tc_name_obj, str) else None
            content = result.error if result.content is None else result.content
            messages.append(
                BatchMessage(
                    role="tool",
                    content=content or "",
                    name=tc_name or "unknown",
                    tool_call_id=result.id,
                )
            )

        state_obj = json_compatible_value(self.config.state)
        if not isinstance(state_obj, dict):
            raise ValueError("Expected chain state to be a JSON-compatible object.")
        state: dict[str, object] = {}
        for key, value in state_obj.items():
            state[str(key)] = value
        return BatchSubmitRequest(
            request_id=f"{self.config.agent_id}-{uuid4()}",
            agent_id=self.config.agent_id,
            step_id=submit_request.step_id,
            step_kind=submit_request.step_kind,
            output_schema=submit_request.output_schema,
            model=submit_request.model,
            messages=messages,
            response_format=submit_request.response_format,
            tool_ids=list(self.config.tool_ids),
            tools=submit_request.tools,
            tool_use_enabled=submit_request.tool_use_enabled,
            bedrock_model_id=submit_request.bedrock_model_id,
            context=BatchAgentContext(
                input_text=self.config.run_input.text,
                state=state,
                memory=dict(self.config.memory),
                safe_dir=self.config.loaded.spec.safe_dir,
                extra_tools=list(self.config.extra_tools or []),
            ),
        )

    async def _call_batch_handler(
        self, batch_request: BatchSubmitRequest
    ) -> BatchImportRequest:
        handler_obj = self.config.batch_call
        if handler_obj is None:
            return await self._local_batch_call(batch_request)
        if not callable(handler_obj):
            raise ValueError("config.batch_call must be callable when provided.")
        response = handler_obj(batch_request)
        if isinstance(response, BatchImportRequest):
            return response
        if inspect.isawaitable(response):
            awaited = await response
            if isinstance(awaited, BatchImportRequest):
                return awaited
            return BatchImportRequest.model_validate(awaited)
        return BatchImportRequest.model_validate(response)

    def import_outcome(self, *, batch_import: BatchImportRequest) -> BatchImportOutcome:
        payload = batch_import.payload
        state_obj = payload.get("state")
        if not isinstance(state_obj, str):
            raise ValueError("Batch import payload is missing string field 'state'.")
        if state_obj == "error":
            message_obj = payload.get("message")
            if not isinstance(message_obj, str):
                raise ValueError(
                    "Error batch payload is missing string field 'message'."
                )
            mapped_error = self._map_model_error_message(message_obj)
            if mapped_error is not None:
                raise mapped_error
            raise ValueError(message_obj)
        if state_obj == "completed":
            if "output" not in payload:
                raise ValueError("Completed batch payload is missing 'output'.")
            return BatchImportOutcome(result=_as_agent_result(payload["output"]))
        if state_obj == "submit_next":
            next_request_obj = payload.get("next_request")
            if not isinstance(next_request_obj, dict):
                raise ValueError(
                    "submit_next batch payload is missing object field 'next_request'."
                )
            next_request = BatchSubmitRequest.model_validate(next_request_obj)
            return BatchImportOutcome(next_request=next_request)
        if state_obj == "tool_use":
            tool_calls_obj = payload.get("tool_calls")
            if not isinstance(tool_calls_obj, list):
                raise ValueError(
                    "tool_use batch payload is missing list field 'tool_calls'."
                )
            raw: list[dict[str, object]] = []
            for tc in tool_calls_obj:
                if isinstance(tc, dict):
                    raw.append({str(k): v for k, v in tc.items()})
            tool_calls = normalize_tool_calls(raw)
            pending_submit_request: BatchSubmitRequest | None = None
            submit_request_obj = payload.get("submit_request")
            if isinstance(submit_request_obj, dict):
                pending_submit_request = BatchSubmitRequest.model_validate(
                    submit_request_obj
                )
            return BatchImportOutcome(
                tool_calls=tool_calls,
                pending_submit_request=pending_submit_request,
            )
        raise ValueError(f"Unsupported batch import state: {state_obj}")

    def _map_model_error_message(self, message: str) -> QuickAgentException | None:
        if "does not support tools" in message:
            return QuickAgentToolsNotSupportedException(
                model_name=self.config.model_spec.model_name,
                message=message,
            )
        if "does not support chat" in message:
            return QuickAgentChatNotSupportedException(
                model_name=self.config.model_spec.model_name,
                message=message,
            )
        return None

    async def _local_batch_call(
        self, batch_request: BatchSubmitRequest
    ) -> BatchImportRequest:
        prefix = "AgentExecutor._local_batch_call"
        api_key_env = self.config.model_spec.api_key_env
        api_key = os.environ.get(api_key_env, "noop")
        logger.debug(f"{prefix}: api_key_env={api_key_env}")
        client = self.context.build_client(self.config)
        messages: list[ChatCompletionMessageParam] = []
        convert_null = self.config.model_spec.convert_null
        if convert_null is None:
            convert_null = _should_convert_null(self.config.model_spec.base_url)
        for batch_message in batch_request.messages:
            if batch_message.role == "system":
                system_message: ChatCompletionSystemMessageParam = {
                    "role": "system",
                    "content": batch_message.content or "",
                }
                messages.append(system_message)
            elif batch_message.role == "assistant":
                assistant_message: ChatCompletionAssistantMessageParam = {
                    "role": "assistant",
                }
                if batch_message.content is not None:
                    assistant_message["content"] = batch_message.content
                elif convert_null:
                    assistant_message["content"] = ""
                if batch_message.tool_calls is not None:
                    assistant_message["tool_calls"] = batch_message.tool_calls  # type: ignore[typeddict-item]
                messages.append(assistant_message)
            elif batch_message.role == "tool":
                tool_message: ChatCompletionToolMessageParam = {
                    "role": "tool",
                    "content": batch_message.content or "",
                    "tool_call_id": batch_message.tool_call_id or "",
                }
                messages.append(tool_message)
            else:
                user_message: ChatCompletionUserMessageParam = {
                    "role": "user",
                    "content": batch_message.content or "",
                }
                messages.append(user_message)
        extra_body: dict[str, JsonValue] | None = None
        if self.context.model_settings_json is not None:
            extra_body_obj = self.context.model_settings_json.extra_body
            if isinstance(extra_body_obj, dict):
                extra_body = extra_body_obj
        if batch_request.response_format is not None:
            if extra_body is None:
                extra_body = {}
            extra_body["response_format"] = batch_request.response_format
        tools_payload: list[ChatCompletionToolUnionParam] = []
        if self.config.toolset is not None and self.config.tool_ids:
            for tool in self.config.toolset.tools.values():
                raw_params = tool.function_schema.json_schema
                params: dict[str, Any] = dict(raw_params)
                if takes_agent_state(tool.function):
                    stripped = strip_agent_state_from_schema(raw_params)
                    params = dict(stripped)
                function_def = FunctionDefinition(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=params,
                )
                tools_payload.append(
                    ChatCompletionFunctionToolParam(
                        {"type": "function", "function": function_def}
                    )
                )
        try:
            response_format = (
                typing.cast(ResponseFormat, batch_request.response_format)
                if batch_request.response_format is not None
                else openai.omit
            )
            temperature = batch_request.model.temperature
            max_completion_tokens = batch_request.model.max_completion_tokens
            model_settings = self.context.model_settings_json
            if model_settings.max_completion_tokens is not openai.omit:
                max_completion_tokens = model_settings.max_completion_tokens
            if model_settings.temperature is not openai.omit:
                temperature = model_settings.temperature
            response = await client.chat.completions.create(
                model=batch_request.model.model_name,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                response_format=response_format,
                tools=tools_payload or openai.omit,
                extra_body=extra_body,
                extra_headers=model_settings.extra_headers,
                timeout=model_settings.timeout,
                top_p=model_settings.top_p,
                presence_penalty=model_settings.presence_penalty,
                frequency_penalty=model_settings.frequency_penalty,
                logit_bias=model_settings.logit_bias,
                stop=model_settings.stop,
                seed=model_settings.seed,
                parallel_tool_calls=model_settings.parallel_tool_calls,
            )
        except openai.APIStatusError as error:
            body_obj = error.body
            error_message = str(error)
            if isinstance(body_obj, dict):
                body_message = body_obj.get("message")
                if isinstance(body_message, str):
                    error_message = body_message
            elif isinstance(body_obj, str):
                error_message = body_obj
            mapped_error = self._map_model_error_message(error_message)
            if mapped_error is not None:
                raise mapped_error from error
            raise
        self._capture_metrics(usage=getattr(response, "usage", {}), response=response)
        if not response.choices:
            raise ValueError("Model returned no completion choices.")
        message_obj = response.choices[0].message
        content_obj = message_obj.content
        tool_calls = getattr(message_obj, "tool_calls", None)

        if not content_obj and not tool_calls:
            refusal_obj = getattr(message_obj, "refusal", None)
            if refusal_obj:
                raise ValueError(f"Model refused response: {refusal_obj}")
            raise ValueError("Model returned an empty response.")

        if tool_calls:
            return BatchImportRequest(
                request_id=batch_request.request_id,
                provider_job_id=getattr(response, "id", None),
                payload={
                    "state": "tool_use",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        }
                        for tool_call in tool_calls
                    ],
                    "submit_request": batch_request.model_dump(mode="json"),
                },
            )
        return BatchImportRequest(
            request_id=batch_request.request_id,
            provider_job_id=getattr(response, "id", None),
            payload={
                "state": "completed",
                "output": content_obj,
            },
        )

    async def _execute_batch_request(
        self, *, batch_request: BatchSubmitRequest, schema_cls: Type[BaseModel] | None
    ) -> BaseModel | str:
        request = batch_request
        while True:
            batch_import = await self._call_batch_handler(request)
            outcome = self.import_outcome(batch_import=batch_import)
            if outcome.tool_calls is not None:
                pending = outcome.pending_submit_request
                if pending is None:
                    raise ValueError(
                        "tool_use outcome is missing pending_submit_request."
                    )
                executed = await self._execute_tool_calls(outcome.tool_calls)
                request = self._build_next_request_with_tool_results(
                    tool_calls=outcome.tool_calls,
                    executed=executed,
                    submit_request=pending,
                )
                continue
            if outcome.next_request is not None:
                request = outcome.next_request
                continue
            raw_result = outcome.result
            if raw_result is None:
                raise ValueError("Batch import outcome did not include a final result.")
            if schema_cls is None:
                if isinstance(raw_result, str):
                    return raw_result
                raise ValueError("Text step expected a string output.")
            return parse_structured_result(raw_result, schema_cls)
