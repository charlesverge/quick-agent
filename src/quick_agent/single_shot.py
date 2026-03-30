"""Single-shot execution strategies."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Type

import openai
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.settings import ModelSettings

from quick_agent.exceptions import (
    QuickAgentChatNotSupportedException,
)
from quick_agent.json_utils import extract_first_json_object
from quick_agent.types import AgentResult

if TYPE_CHECKING:
    from quick_agent.quick_agent import QuickAgent


def _single_shot_messages(
    *,
    instructions: str | None,
    system_prompt: str | list[str],
    user_prompt: str,
) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []
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
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": "\n".join(system_parts),
        }
        messages.append(system_message)
    user_message: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": user_prompt,
    }
    messages.append(user_message)
    return messages


def _extract_openai_error_message(error: openai.APIStatusError) -> str:
    body = error.body
    if isinstance(body, dict):
        body_message = body.get("message")
        if isinstance(body_message, str):
            return body_message
    if isinstance(body, str):
        return body
    return str(error)


def _parse_structured_result(
    raw_output: AgentResult, schema_cls: Type[BaseModel]
) -> BaseModel:
    if isinstance(raw_output, BaseModel):
        return raw_output
    try:
        if isinstance(raw_output, str):
            return schema_cls.model_validate_json(raw_output)
        return schema_cls.model_validate(raw_output)
    except ValidationError:
        if isinstance(raw_output, str):
            extracted = extract_first_json_object(raw_output)
            return schema_cls.model_validate_json(extracted)
        raise


async def _run_single_shot_text_via_pydantic_ai(
    agent: QuickAgent,
    *,
    user_prompt: str,
    instructions: str | None,
    system_prompt: str | list[str],
    model_settings: ModelSettings | None,
) -> str:
    toolsets = agent._toolsets_for_run()
    runner = Agent(
        agent.model,
        instructions=instructions,
        system_prompt=system_prompt,
        toolsets=toolsets,
        output_type=str,
        model_settings=model_settings,
    )
    try:
        result = await runner.run(user_prompt)
    except ModelHTTPError as error:
        mapped_error = agent._map_model_http_error(error)
        if mapped_error is not None:
            raise mapped_error from error
        raise error
    agent._capture_pydantic_ai_metrics(result)
    return result.output


async def _run_single_shot_structured_via_pydantic_ai(
    agent: QuickAgent,
    *,
    schema_cls: Type[BaseModel],
    user_prompt: str,
    instructions: str | None,
    system_prompt: str | list[str],
    model_settings: ModelSettings | None,
) -> BaseModel:
    toolsets = agent._toolsets_for_run()
    runner = Agent(
        agent.model,
        instructions=instructions,
        system_prompt=system_prompt,
        toolsets=toolsets,
        output_type=schema_cls,
        model_settings=model_settings,
    )
    try:
        result = await runner.run(user_prompt)
    except ModelHTTPError as error:
        mapped_error = agent._map_model_http_error(error)
        if mapped_error is not None:
            raise mapped_error from error
        raise error
    agent._capture_pydantic_ai_metrics(result)
    return _parse_structured_result(result.output, schema_cls)


async def _run_single_shot_structured_via_openai_sdk(
    agent: QuickAgent,
    *,
    schema_cls: Type[BaseModel],
    user_prompt: str,
    instructions: str | None,
    system_prompt: str | list[str],
    model_settings: ModelSettings | None,
) -> BaseModel:
    if agent.has_tools():
        raise ValueError(
            "output.output_schema does not support tools in single-shot mode."
        )
    api_key = os.environ.get(agent.model_spec.api_key_env, "noop")
    timeout_seconds = agent.model_spec.timeout_seconds
    client = agent.client or openai.AsyncOpenAI(
        api_key=api_key,
        base_url=agent.model_spec.base_url,
        timeout=timeout_seconds,
        http_client=agent._http_client,
    )
    messages = _single_shot_messages(
        instructions=instructions,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    # NOTE: Work around pydantic-ai structured single-shot behavior that can trigger tool-style output paths.
    # Track upstream status: https://github.com/pydantic/pydantic-ai/pull/4160
    response = None
    try:
        response_format: ResponseFormatJSONSchema = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_cls.__name__,
                "schema": schema_cls.model_json_schema(),
                "strict": True,
            },
        }
        extra_body = None
        if model_settings is not None:
            extra_body = model_settings.get("extra_body")

        response = await client.chat.completions.create(
            model=agent.model_spec.model_name,
            messages=messages,
            temperature=agent.model_spec.temperature,
            max_completion_tokens=agent.model_spec.max_completion_tokens,
            response_format=response_format,
            extra_body=extra_body,
        )
    except openai.APIStatusError as error:
        message = _extract_openai_error_message(error)
        if "does not support chat" in message:
            raise QuickAgentChatNotSupportedException(
                model_name=agent.model_spec.model_name, message=message
            ) from error
        raise
    if response is not None:
        agent._capture_openai_sdk_metrics(response)
    if response is None or not response.choices:
        raise ValueError("Model returned no completion choices.")
    message_obj = response.choices[0].message
    content_obj = message_obj.content
    if not content_obj or not content_obj.strip():
        refusal_obj = message_obj.refusal
        if refusal_obj:
            raise ValueError(f"Model refused structured response: {refusal_obj}")
        raise ValueError("Model returned an empty structured response.")
    return _parse_structured_result(content_obj, schema_cls)


async def run_single_shot(
    agent: QuickAgent, *, schema_cls: Type[BaseModel] | None
) -> BaseModel | str:
    user_prompt = agent._build_single_shot_prompt()
    instructions = agent.loaded.instructions
    system_prompt = agent.loaded.system_prompt
    model_settings = agent.model_settings_json
    if schema_cls is not None:
        model_settings = agent._build_structured_model_settings(schema_cls=schema_cls)

    agent._record_llm_request(
        call_site="run_single_shot",
        step_id=None,
        step_kind="single_shot",
        output_schema=agent.loaded.spec.output.output_schema,
        instructions=instructions,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model_settings=model_settings,
    )

    if schema_cls is None:
        return await _run_single_shot_text_via_pydantic_ai(
            agent,
            user_prompt=user_prompt,
            instructions=instructions,
            system_prompt=system_prompt,
            model_settings=model_settings,
        )

    if agent.loaded.spec.single_shot_use_pydantic_ai:
        return await _run_single_shot_structured_via_pydantic_ai(
            agent,
            schema_cls=schema_cls,
            user_prompt=user_prompt,
            instructions=instructions,
            system_prompt=system_prompt,
            model_settings=model_settings,
        )

    return await _run_single_shot_structured_via_openai_sdk(
        agent,
        schema_cls=schema_cls,
        user_prompt=user_prompt,
        instructions=instructions,
        system_prompt=system_prompt,
        model_settings=model_settings,
    )
