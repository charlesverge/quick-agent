from __future__ import annotations

from dataclasses import dataclass

from openai.types.chat import ChatCompletionAssistantMessageParam
from pydantic_ai.models.openai import OpenAIChatModel


# ---------------------------------------------------------------------------
# Ollama-safe model subclass: patches content=None → content="" in assistant
# messages to work around Ollama rejecting null content as <nil>.
# See docs/tool_mode.md for detailed reasoning.
# ---------------------------------------------------------------------------
class OllamaSafeChatModel(OpenAIChatModel):
    """OpenAIChatModel that replaces content=None with content='' in assistant
    messages, preventing Ollama's 'invalid message content type: <nil>' error.
    """

    @dataclass
    class _MapModelResponseContext(OpenAIChatModel._MapModelResponseContext):
        def _into_message_param(self) -> ChatCompletionAssistantMessageParam:
            message_param = super()._into_message_param()
            if message_param.get("content") is None:
                message_param["content"] = ""
            return message_param
