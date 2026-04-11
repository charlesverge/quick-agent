"""Pydantic models for batch request generation and import parsing."""

from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    Field,
    JsonValue,
    SkipValidation,
    field_serializer,
    model_validator,
)
from quick_agent.types import AgentResult


class BatchMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    tool_call_id: str | None = None


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
    tool_ids: list[str] = Field(default_factory=list)
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

    def _split_messages(self) -> tuple[list[str], list[str], list[str]]:
        system_items: list[str] = []
        user_items: list[str] = []
        assistant_items: list[str] = []
        for message in self.messages:
            if message.role == "system":
                system_items.append(message.content)
                continue
            if message.role == "user":
                user_items.append(message.content)
                continue
            if message.role == "assistant":
                assistant_items.append(message.content)
                continue
            raise ValueError(f"Unsupported Bedrock batch role: {message.role}")
        return system_items, user_items, assistant_items

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

    def _build_converse_jsonl_line(self) -> dict[str, object]:
        system_items, user_items, assistant_items = self._split_messages()
        messages: list[ConverseMessage] = []
        for item in user_items:
            messages.append(
                ConverseMessage(role="user", content=[ConverseContent(text=item)])
            )
        for item in assistant_items:
            messages.append(
                ConverseMessage(role="assistant", content=[ConverseContent(text=item)])
            )
        model_input: dict[str, object] = {"messages": messages}
        if system_items:
            model_input["system"] = [
                ConverseContent(text=item) for item in system_items
            ]
        inference_config = self._inference_config()
        if inference_config:
            model_input["inferenceConfig"] = inference_config
        if self.model.extra_body is not None:
            for key, value in self.model.extra_body.items():
                if key in ("inferenceConfig", "bedrock_request_mode"):
                    continue
                if key == "response_format":
                    if "outputConfig" not in model_input:
                        model_input["outputConfig"] = {"textFormat": value}
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
        if self.response_format is not None:
            if "outputConfig" not in model_input:
                model_input["outputConfig"] = {"textFormat": self.response_format}
        record = ConverseBatchRecord(
            recordId=self.request_id,
            modelInput=ConverseModelInput.model_validate(model_input),
        )
        return record.model_dump(mode="json", exclude_none=True)

    def _build_open_weight_invoke_jsonl_line(self) -> dict[str, object]:
        system_items, user_items, assistant_items = self._split_messages()
        messages: list[InvokeModelMessage] = []
        for item in system_items:
            messages.append(InvokeModelMessage(role="system", content=item))
        for item in user_items:
            messages.append(InvokeModelMessage(role="user", content=item))
        for item in assistant_items:
            messages.append(InvokeModelMessage(role="assistant", content=item))
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
        record = InvokeBatchRecord(
            recordId=self.request_id,
            modelInput=InvokeModelInput.model_validate(model_input),
        )
        return record.model_dump(mode="json", exclude_none=True)

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
        return row

    @property
    def jsonl_line(self) -> dict[str, object]:
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
    final_result: SkipValidation[AgentResult] | None = None
    next_submit_request: BatchSubmitRequest | None = None

    @field_serializer("final_result", when_used="json")
    def serialize_final_result(self, value: AgentResult | None) -> object:
        return _serialize_agent_result(value)

    @model_validator(mode="after")
    def validate_outcome(self) -> "BatchImportOutcome":
        has_result = self.final_result is not None
        has_next = self.next_submit_request is not None
        if has_result == has_next:
            raise ValueError(
                "BatchImportOutcome must include exactly one of final_result or next_submit_request."
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
