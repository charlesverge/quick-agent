"""Single-shot execution strategies."""

from __future__ import annotations

import json
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
from pydantic import BaseModel

from quick_agent.agent_utils import parse_structured_result
from quick_agent.json_utils import json_compatible_value
from quick_agent.exceptions import (
    QuickAgentChatNotSupportedException,
)

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


async def _run_single_shot_text_via_openai_sdk(
    agent: QuickAgent,
    *,
    user_prompt: str,
    instructions: str | None,
    system_prompt: str | list[str],
    model_settings: dict[str, object] | None,
) -> str:
    from quick_agent.executor import _should_convert_null

    toolsets = agent._toolsets_for_run()
    tools: list[dict[str, object]] = []
    for ts in toolsets:
        for tool in ts.tools.values():
            tools.append(tool.function_schema.to_openai_tool())
    client = agent._executor.context.build_client(agent._executor.config)
    messages = _single_shot_messages(
        instructions=instructions,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    convert_null = agent.model_spec.convert_null
    if convert_null is None:
        convert_null = _should_convert_null(agent.model_spec.base_url)
    max_iterations = 10
    for _ in range(max_iterations):
        extra_body: dict[str, object] | None = None
        if model_settings is not None:
            extra_body = model_settings.get("extra_body")
        response = await client.chat.completions.create(
            model=agent.model_spec.model_name,
            messages=messages,
            tools=tools if tools else None,
            temperature=agent.model_spec.temperature,
            max_completion_tokens=agent.model_spec.max_completion_tokens,
            extra_body=extra_body,
        )
        agent._capture_openai_sdk_metrics(response)
        if not response.choices:
            raise ValueError("Model returned no completion choices.")
        message_obj = response.choices[0].message
        if message_obj.content is None and convert_null:
            message_dict: ChatCompletionMessageParam = {
                "role": "assistant",
                "content": "",
            }
        else:
            message_dict = message_obj.model_dump(exclude_unset=True)
        messages.append(message_dict)
        if not message_obj.tool_calls:
            content = message_obj.content
            return content if content else ""
        for tool_call in message_obj.tool_calls:
            tool_name = tool_call.function.name
            tool_args_str = tool_call.function.arguments
            tool_args: dict[str, object] = {}
            if tool_args_str:
                tool_args = json.loads(tool_args_str)
            for ts in toolsets:
                if tool_name in ts.tools:
                    result = ts.tools[tool_name].function(**tool_args)
                    result_json = json_compatible_value(result)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result_json),
                        }
                    )
                    break
    raise ValueError("Tool call loop exceeded max iterations")


async def _run_single_shot_structured_via_openai_sdk(
    agent: QuickAgent,
    *,
    schema_cls: Type[BaseModel],
    user_prompt: str,
    instructions: str | None,
    system_prompt: str | list[str],
    model_settings: dict[str, object] | None,
) -> BaseModel:
    if agent.has_tools():
        raise ValueError(
            "output.output_schema does not support tools in single-shot mode."
        )
    client = agent._executor.context.build_client(agent._executor.config)
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
    return parse_structured_result(content_obj, schema_cls)


async def run_single_shot(
    agent: QuickAgent, *, schema_cls: Type[BaseModel] | None
) -> BaseModel | str:
    user_prompt = agent._build_single_shot_prompt()
    instructions = agent.loaded.instructions
    system_prompt = agent.loaded.system_prompt
    model_settings = agent._executor.context.model_settings_json
    if schema_cls is not None:
        model_settings = agent._executor.context.build_structured_model_settings(
            schema_cls=schema_cls
        )

    agent._recorder._record_llm_request(
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
        return await _run_single_shot_text_via_openai_sdk(
            agent,
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
