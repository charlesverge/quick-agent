"""Pydantic models for batch request generation and import parsing."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    SkipValidation,
    field_serializer,
    model_validator,
)

from quick_agent.json_utils import validate_bedrock_schema
from quick_agent.models.chain_step_spec import ToolChoice
from quick_agent.types import AgentResult


class BatchToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: dict[str, JsonValue]
    strict: bool = True


class BatchMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, JsonValue]] | None = None


class BatchModelConfig(BaseModel):
    provider: str
    base_url: str
    model_name: str
    temperature: float | None = None
    max_completion_tokens: int | None = None
    extra_headers: dict[str, str] | None = None
    extra_body: dict[str, JsonValue] | None = None
    bedrock_request_mode: (
        Literal["converse", "anthropic_invoke", "open_weight_invoke"] | None
    ) = None


class BatchAgentContext(BaseModel):
    input_text: str = ""
    state: dict[str, object] = Field(default_factory=dict)
    memory: dict[str, object] = Field(default_factory=dict)
    safe_dir: str | None = None
    extra_tools: list[str] = Field(default_factory=list)


class ConverseContent(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ConverseMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: list[ConverseContent]


class ConverseModelInput(BaseModel):
    messages: list[ConverseMessage]
    system: list[ConverseContent] | None = None
    inferenceConfig: dict[str, object] | None = None
    additionalModelRequestFields: JsonValue | None = None
    additionalModelResponseFieldPaths: list[str] | None = None
    guardrailConfig: dict[str, JsonValue] | None = None
    outputConfig: dict[str, JsonValue] | None = None
    performanceConfig: dict[str, JsonValue] | None = None
    promptVariables: dict[str, JsonValue] | None = None
    requestMetadata: dict[str, str] | None = None
    serviceTier: dict[str, JsonValue] | None = None
    toolConfig: dict[str, JsonValue] | None = None


class ConverseBatchRecord(BaseModel):
    recordId: str
    modelInput: ConverseModelInput


class InvokeModelMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class InvokeModelInput(BaseModel):
    messages: list[InvokeModelMessage]
    inferenceConfig: dict[str, object] | None = None
    response_format: dict[str, JsonValue] | None = None
    output_config: dict[str, JsonValue] | None = None
    additionalModelRequestFields: JsonValue | None = None
    additionalModelResponseFieldPaths: list[str] | None = None
    guardrailConfig: dict[str, JsonValue] | None = None
    outputConfig: dict[str, JsonValue] | None = None
    performanceConfig: dict[str, JsonValue] | None = None
    promptVariables: dict[str, JsonValue] | None = None
    requestMetadata: dict[str, str] | None = None
    serviceTier: dict[str, JsonValue] | None = None
    toolConfig: dict[str, JsonValue] | None = None
    anthropic_version: str | None = None


class InvokeBatchRecord(BaseModel):
    recordId: str
    modelInput: InvokeModelInput


class BatchSubmitRequest(BaseModel):
    request_id: str
    agent_id: str
    step_id: str | None
    step_kind: str
    output_schema: str | None = None
    model: BatchModelConfig
    messages: list[BatchMessage]
    response_format: dict[str, JsonValue] | None = None
    tool_choice: ToolChoice | None = None
    max_tool_calls: int = Field(default=3, ge=1)
    tool_ids: list[str] = Field(default_factory=list)
    tools: list[BatchToolDefinition] | None = None
    tool_use_enabled: bool = False
    response_as_tool: bool = False
    final_result_tool_enabled: bool = False
    bedrock_model_id: str | None = None
    context: BatchAgentContext = Field(default_factory=BatchAgentContext)

    def _inference_config(self) -> dict[str, object]:
        inference_config: dict[str, object] = {}
        if self.model.max_completion_tokens is not None:
            inference_config["maxTokens"] = self.model.max_completion_tokens
        if self.model.temperature is not None:
            inference_config["temperature"] = self.model.temperature
        if self.model.extra_body is not None:
            inference_obj = self.model.extra_body.get("inferenceConfig")
            if isinstance(inference_obj, dict):
                for key, value in inference_obj.items():
                    if key == "max_new_tokens":
                        inference_config["maxTokens"] = value
                        continue
                    if key == "top_p":
                        inference_config["topP"] = value
                        continue
                    if key == "stop_sequences":
                        inference_config["stopSequences"] = value
                        continue
                    inference_config[key] = value
        return inference_config

    def _resolve_bedrock_request_mode(
        self,
    ) -> Literal["converse", "anthropic_invoke", "open_weight_invoke"]:
        model_mode = self.model.bedrock_request_mode
        if model_mode is not None:
            return model_mode
        if self.model.extra_body is not None:
            mode_obj = self.model.extra_body.get("bedrock_request_mode")
            if isinstance(mode_obj, str):
                if mode_obj == "converse":
                    return "converse"
                if mode_obj == "anthropic_invoke":
                    return "anthropic_invoke"
                if mode_obj == "open_weight_invoke":
                    return "open_weight_invoke"
            if "anthropic_version" in self.model.extra_body:
                return "anthropic_invoke"
            if isinstance(self.model.extra_body.get("response_format"), dict):
                return "open_weight_invoke"
        model_name = self.model.model_name.lower()
        if model_name.startswith("anthropic."):
            return "anthropic_invoke"
        if "qwen" in model_name:
            return "open_weight_invoke"
        if self.response_format is not None:
            return "open_weight_invoke"
        return "converse"

    def validate_bedrock_model(self, bedrock_model_id: str) -> None:
        actual = self._resolve_bedrock_request_mode()
        check = self.model_copy(
            update={
                "model": self.model.model_copy(
                    update={
                        "model_name": bedrock_model_id,
                        "bedrock_request_mode": None,
                        "extra_body": None,
                    }
                )
            }
        )
        required = check._resolve_bedrock_request_mode()
        if actual != required:
            raise ValueError(
                f"agent_id='{self.agent_id}' model_name='{self.model.model_name}' resolves to '{actual}' format "
                f"but bedrock_model_id='{bedrock_model_id}' requires '{required}' format. "
                f"Update the agent model config to match the deployed Bedrock model."
            )

    def openai_tool_choice(self) -> JsonValue | None:
        if self.tool_choice is None:
            return None
        if self.tool_choice.type == "function" and self.tool_choice.name is not None:
            return {
                "type": "function",
                "function": {"name": self.tool_choice.name},
            }
        mode = self.tool_choice.mode
        if mode == "required":
            return "required"
        if mode == "none":
            return "none"
        return None

    def bedrock_converse_tool_choice(self) -> dict[str, object] | None:
        if self.tool_choice is None:
            return None
        if self.tool_choice.type == "function" and self.tool_choice.name is not None:
            return {"tool": {"name": self.tool_choice.name}}
        mode = self.tool_choice.mode
        if mode in ("required", "any"):
            return {"any": {}}
        return None

    def bedrock_invoke_tool_choice(self) -> JsonValue | None:
        if self.tool_choice is None:
            return None
        if self.tool_choice.type == "function" and self.tool_choice.name is not None:
            return {
                "type": "function",
                "function": {"name": self.tool_choice.name},
            }
        mode = self.tool_choice.mode
        if mode == "required":
            return "required"
        if mode == "any":
            return "required"
        if mode == "none":
            return "none"
        return None

    def _bedrock_response_schema(self) -> dict[str, object] | None:
        response_format = self.response_format
        if response_format is None and self.model.extra_body is not None:
            response_format_obj = self.model.extra_body.get("response_format")
            if isinstance(response_format_obj, dict):
                response_format = response_format_obj
        if response_format is None:
            return None
        if response_format.get("type") != "json_schema":
            raise ValueError(
                "Bedrock structured output requires response_format.type='json_schema'."
            )
        json_schema_obj = response_format.get("json_schema")
        if isinstance(json_schema_obj, dict):
            schema_obj = json_schema_obj.get("schema")
            if isinstance(schema_obj, dict):
                schema: dict[str, object] = {}
                for key, value in schema_obj.items():
                    schema[str(key)] = value
                return schema
        structure_obj = response_format.get("structure")
        if not isinstance(structure_obj, dict):
            raise ValueError(
                "Bedrock structured output requires response_format.json_schema or response_format.structure.jsonSchema."
            )
        structure_schema = structure_obj.get("jsonSchema")
        if not isinstance(structure_schema, dict):
            raise ValueError(
                "Bedrock structured output requires response_format.structure.jsonSchema."
            )
        raw_schema = structure_schema.get("schema")
        if isinstance(raw_schema, dict):
            schema = {}
            for key, value in raw_schema.items():
                schema[str(key)] = value
            return schema
        if not isinstance(raw_schema, str):
            raise ValueError(
                "Bedrock structured output requires jsonSchema.schema to be a JSON string or object."
            )
        parsed = json.loads(raw_schema)
        if not isinstance(parsed, dict):
            raise ValueError(
                "Bedrock structured output requires jsonSchema.schema to decode to an object."
            )
        schema = {}
        for key, value in parsed.items():
            schema[str(key)] = value
        return schema

    def _validate_bedrock_tools(self) -> None:
        if self.tools is None:
            return
        for tool in self.tools:
            if tool.strict is not True:
                raise ValueError(f"Bedrock tool '{tool.name}' must set strict=true.")
            schema: dict[str, object] = {}
            for key, value in tool.input_schema.items():
                schema[str(key)] = value
            validate_bedrock_schema(
                schema,
                context=f"Bedrock tool schema for '{tool.name}'",
            )

    def tool_call_rounds(self) -> int:
        count = 0
        for message in self.messages:
            if message.role != "assistant" or message.tool_calls is None:
                continue
            if len(message.tool_calls) > 0:
                count += 1
        return count

    def _build_converse_jsonl_line(self) -> dict[str, object]:
        is_noop = self.request_id.startswith("noop-") or ":noop:" in self.request_id
        converse_msgs: list[dict[str, object]] = []
        system_blocks: list[dict[str, object]] = []
        for message in self.messages:
            if message.role == "system":
                if message.content and not is_noop:
                    system_blocks.append({"text": message.content})
                continue
            if message.role == "user":
                content_blocks: list[dict[str, object]] = [
                    {"text": message.content or ""}
                ]
                converse_msgs.append({"role": "user", "content": content_blocks})
                continue
            if message.role == "tool":
                tool_result: dict[str, object] = {
                    "toolUseId": message.tool_call_id,
                    "content": [{"text": message.content or ""}],
                    "status": "success" if not message.name or True else "error",
                }
                converse_msgs.append(
                    {"role": "user", "content": [{"toolResult": tool_result}]}
                )
                continue
            if message.role == "assistant":
                assistant_content_blocks: list[dict[str, object]] = []
                if message.content:
                    assistant_content_blocks.append({"text": message.content})
                if message.tool_calls:
                    for tc in message.tool_calls:
                        func_obj = tc.get("function")
                        tc_id = tc.get("id")
                        tc_name: object = None
                        tc_input: dict[str, JsonValue] = {}
                        if isinstance(func_obj, dict):
                            tc_name = func_obj.get("name")
                            args_val = func_obj.get("arguments")
                            if isinstance(args_val, str):
                                try:
                                    parsed = json.loads(args_val)
                                    if isinstance(parsed, dict):
                                        tc_input = parsed
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            elif isinstance(args_val, dict):
                                tc_input = args_val
                        assistant_content_blocks.append(
                            {
                                "toolUse": {
                                    "toolUseId": tc_id,
                                    "name": tc_name,
                                    "input": tc_input,
                                }
                            }
                        )
                if not assistant_content_blocks:
                    assistant_content_blocks.append({"text": ""})
                converse_msgs.append(
                    {"role": "assistant", "content": assistant_content_blocks}
                )
                continue
            raise ValueError(f"Unsupported Bedrock batch role: {message.role}")
        model_input: dict[str, object] = {"messages": converse_msgs}
        if system_blocks:
            model_input["system"] = system_blocks
        inference_config = self._inference_config()
        if inference_config:
            model_input["inferenceConfig"] = inference_config
        if self.model.extra_body is not None:
            for key, value in self.model.extra_body.items():
                if key in ("inferenceConfig", "bedrock_request_mode"):
                    continue
                if key == "response_format":
                    if is_noop:
                        continue
                    converted_text_format = value
                    if (
                        isinstance(value, dict)
                        and value.get("type") == "json_schema"
                        and "json_schema" in value
                        and "structure" not in value
                    ):
                        json_schema_obj = value.get("json_schema")
                        if isinstance(json_schema_obj, dict):
                            schema_obj = json_schema_obj.get("schema")
                            if isinstance(schema_obj, dict):
                                schema_obj = json.dumps(schema_obj)
                            converted_text_format = {
                                "type": "json_schema",
                                "structure": {
                                    "jsonSchema": {
                                        "name": json_schema_obj.get("name"),
                                        "description": json_schema_obj.get(
                                            "description"
                                        ),
                                        "schema": schema_obj,
                                    }
                                },
                            }
                    if "outputConfig" not in model_input:
                        model_input["outputConfig"] = {
                            "textFormat": converted_text_format
                        }
                    continue
                if key == "outputConfig" and is_noop:
                    continue
                if key in (
                    "additionalModelRequestFields",
                    "additionalModelResponseFieldPaths",
                    "guardrailConfig",
                    "outputConfig",
                    "performanceConfig",
                    "promptVariables",
                    "requestMetadata",
                    "serviceTier",
                    "toolConfig",
                ):
                    model_input[key] = value
        if self.response_format is not None and not is_noop:
            text_format: dict[str, JsonValue] = self.response_format
            if (
                self.response_format.get("type") == "json_schema"
                and "json_schema" in self.response_format
                and "structure" not in self.response_format
            ):
                json_schema_obj = self.response_format.get("json_schema")
                if isinstance(json_schema_obj, dict):
                    schema_obj = json_schema_obj.get("schema")
                    if isinstance(schema_obj, dict):
                        schema_obj = json.dumps(schema_obj)
                    text_format = {
                        "type": "json_schema",
                        "structure": {
                            "jsonSchema": {
                                "name": json_schema_obj.get("name"),
                                "description": json_schema_obj.get("schema", {}).get("description", ""),
                                "schema": schema_obj,
                            }
                        },
                    }
            if "outputConfig" not in model_input:
                model_input["outputConfig"] = {"textFormat": text_format}
        if self.tools and "toolConfig" not in model_input:
            tool_specs: list[dict[str, object]] = []
            for tool_def in self.tools:
                tool_specs.append(
                    {
                        "toolSpec": {
                            "name": tool_def.name,
                            "description": tool_def.description,
                            "inputSchema": {"json": tool_def.input_schema},
                            "strict": tool_def.strict,
                        }
                    }
                )
            model_input["toolConfig"] = {"tools": tool_specs}
        converse_tool_choice = self.bedrock_converse_tool_choice()
        if converse_tool_choice is not None:
            tool_config_obj = model_input.get("toolConfig")
            if isinstance(tool_config_obj, dict):
                tool_config_obj["toolChoice"] = converse_tool_choice
            else:
                model_input["toolConfig"] = {"toolChoice": converse_tool_choice}
        return {
            "recordId": self.request_id,
            "modelInput": model_input,
        }

    def _build_open_weight_invoke_jsonl_line(self) -> dict[str, object]:
        messages: list[dict[str, object]] = []
        for message in self.messages:
            if message.role == "system":
                messages.append({"role": "system", "content": message.content or ""})
                continue
            if message.role == "user":
                messages.append({"role": "user", "content": message.content or ""})
                continue
            if message.role == "assistant":
                msg: dict[str, object] = {
                    "role": "assistant",
                    "content": message.content or "",
                }
                if message.tool_calls:
                    msg["tool_calls"] = list(message.tool_calls)
                messages.append(msg)
                continue
            if message.role == "tool":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content or "",
                    }
                )
                continue
            raise ValueError(f"Unsupported batch role: {message.role}")
        model_input: dict[str, object] = {"messages": messages}
        inference_config = self._inference_config()
        if inference_config:
            model_input["inferenceConfig"] = inference_config
        if self.model.extra_body is not None:
            for key, value in self.model.extra_body.items():
                if key in (
                    "inferenceConfig",
                    "bedrock_request_mode",
                    "response_format",
                ):
                    continue
                if key in (
                    "additionalModelRequestFields",
                    "additionalModelResponseFieldPaths",
                    "guardrailConfig",
                    "outputConfig",
                    "performanceConfig",
                    "promptVariables",
                    "requestMetadata",
                    "serviceTier",
                    "toolConfig",
                ):
                    model_input[key] = value
        response_format = self.response_format
        if response_format is None and self.model.extra_body is not None:
            response_format_obj = self.model.extra_body.get("response_format")
            if isinstance(response_format_obj, dict):
                response_format = response_format_obj
        if response_format is not None:
            model_input["response_format"] = response_format
        if self.tools and "toolConfig" not in model_input:
            tool_defs: list[dict[str, object]] = []
            for tool_def in self.tools:
                tool_defs.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool_def.name,
                            "description": tool_def.description,
                            "parameters": tool_def.input_schema,
                            "strict": tool_def.strict,
                        },
                    }
                )
            model_input["tools"] = tool_defs
        invoke_tool_choice = self.bedrock_invoke_tool_choice()
        if invoke_tool_choice is not None:
            model_input["tool_choice"] = invoke_tool_choice
        return {
            "recordId": self.request_id,
            "modelInput": model_input,
        }

    def _build_anthropic_invoke_jsonl_line(self) -> dict[str, object]:
        row = self._build_open_weight_invoke_jsonl_line()
        model_input_obj = row.get("modelInput")
        if not isinstance(model_input_obj, dict):
            raise ValueError("Expected object modelInput for anthropic invoke mode.")
        response_format = self.response_format
        if response_format is None and self.model.extra_body is not None:
            response_format_obj = self.model.extra_body.get("response_format")
            if isinstance(response_format_obj, dict):
                response_format = response_format_obj
        if response_format is not None and "output_config" not in model_input_obj:
            model_input_obj["output_config"] = {"format": response_format}
        if self.model.extra_body is not None:
            anthropic_version_obj = self.model.extra_body.get("anthropic_version")
            if isinstance(anthropic_version_obj, str):
                model_input_obj["anthropic_version"] = anthropic_version_obj
        model_input_obj.pop("response_format", None)
        # Convert tool messages to Anthropic format
        messages_obj = model_input_obj.get("messages")
        if isinstance(messages_obj, list):
            converted: list[dict[str, object]] = []
            for msg in messages_obj:
                if not isinstance(msg, dict):
                    converted.append(msg)
                    continue
                role = msg.get("role")
                if role == "tool":
                    # Convert to Anthropic tool_result format in user message
                    converted.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": msg.get("tool_call_id"),
                                    "content": msg.get("content", ""),
                                }
                            ],
                        }
                    )
                elif role == "assistant" and msg.get("tool_calls"):
                    # Convert tool_calls to Anthropic tool_use format
                    content_blocks: list[dict[str, object]] = []
                    text_content = msg.get("content")
                    if text_content and isinstance(text_content, str):
                        content_blocks.append({"type": "text", "text": text_content})
                    for tc in msg.get("tool_calls", []):
                        func_obj = tc.get("function") if isinstance(tc, dict) else None
                        tc_id = tc.get("id") if isinstance(tc, dict) else None
                        tc_name: object = None
                        tc_input: dict[str, object] = {}
                        if isinstance(func_obj, dict):
                            tc_name = func_obj.get("name")
                            args_val = func_obj.get("arguments")
                            if isinstance(args_val, str):
                                try:
                                    parsed = json.loads(args_val)
                                    if isinstance(parsed, dict):
                                        tc_input = parsed
                                except (json.JSONDecodeError, ValueError):
                                    pass
                            elif isinstance(args_val, dict):
                                tc_input = args_val
                        content_blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": tc_name,
                                "input": tc_input,
                            }
                        )
                    converted.append({"role": "assistant", "content": content_blocks})
                else:
                    converted.append(msg)
            model_input_obj["messages"] = converted
        # Convert tools to Anthropic format
        tools_obj = model_input_obj.get("tools")
        if isinstance(tools_obj, list):
            anthropic_tools: list[dict[str, object]] = []
            for tool in tools_obj:
                if isinstance(tool, dict):
                    func_def = (
                        tool.get("function")
                        if isinstance(tool.get("function"), dict)
                        else {}
                    )
                    anthropic_tools.append(
                        {
                            "name": func_def.get("name")
                            if isinstance(func_def, dict)
                            else None,
                            "description": func_def.get("description")
                            if isinstance(func_def, dict)
                            else None,
                            "input_schema": func_def.get("parameters")
                            if isinstance(func_def, dict)
                            else {},
                            "strict": func_def.get("strict")
                            if isinstance(func_def, dict)
                            else False,
                        }
                    )
            model_input_obj["tools"] = anthropic_tools
        return row

    @property
    def jsonl_line(self) -> dict[str, object]:
        is_noop = self.request_id.startswith("noop-") or ":noop:" in self.request_id
        response_schema = None if is_noop else self._bedrock_response_schema()
        if response_schema is not None:
            validate_bedrock_schema(
                response_schema,
                context=f"Bedrock structured output schema for '{self.request_id}'",
            )
        self._validate_bedrock_tools()
        mode = self._resolve_bedrock_request_mode()
        if mode == "converse":
            return self._build_converse_jsonl_line()
        if mode == "anthropic_invoke":
            return self._build_anthropic_invoke_jsonl_line()
        return self._build_open_weight_invoke_jsonl_line()


class BatchImportRequest(BaseModel):
    request_id: str
    provider_job_id: str | None = None
    payload: dict[str, JsonValue]


class BatchImportOutcome(BaseModel):
    result: SkipValidation[AgentResult] | None = None
    next_request: BatchSubmitRequest | None = None
    tool_calls: list[dict[str, object]] | None = None
    pending_submit_request: BatchSubmitRequest | None = None

    @field_serializer("result", when_used="json")
    def serialize_result(self, value: AgentResult | None) -> object:
        return _serialize_agent_result(value)

    @model_validator(mode="after")
    def validate_outcome(self) -> "BatchImportOutcome":
        if (
            self.result is None
            and self.next_request is None
            and self.tool_calls is None
        ):
            raise ValueError(
                "BatchImportOutcome must include result, next_request, or tool_calls."
            )
        return self


def _serialize_agent_result(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        result_list: list[object] = []
        for item in value:
            result_list.append(_serialize_agent_result(item))
        return result_list
    if isinstance(value, dict):
        result_dict: dict[str, object] = {}
        for key, item in value.items():
            result_dict[key] = _serialize_agent_result(item)
        return result_dict
    raise ValueError(f"Unsupported AgentResult value type: {type(value)}")
