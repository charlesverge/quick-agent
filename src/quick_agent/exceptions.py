"""QuickAgent-specific exceptions."""

from __future__ import annotations

import json
import shlex

import httpx
import openai
from pydantic_ai.exceptions import UnexpectedModelBehavior


class QuickAgentException(Exception):
    """Base exception for QuickAgent runtime failures."""


class QuickAgentToolsNotSupportedException(QuickAgentException):
    """Raised when a model rejects tool usage."""

    def __init__(self, *, model_name: str, message: str) -> None:
        super().__init__(message)
        self.model_name = model_name
        self.message = message


class QuickAgentChatNotSupportedException(QuickAgentException):
    """Raised when a model does not support chat completions."""

    def __init__(self, *, model_name: str, message: str) -> None:
        super().__init__(message)
        self.model_name = model_name
        self.message = message


class QuickAgentUnexpectedModelBehaviorException(QuickAgentException):
    """Raised when model behavior is unexpected with full request/response context."""

    def __init__(
        self,
        *,
        original_exception: UnexpectedModelBehavior,
        request_context: dict[str, object] | None = None,
    ) -> None:
        self.original_exception = original_exception
        self.request_context = request_context
        self.details = self._extract_details(original_exception)
        self.message = f"Unexpected model behavior.\nDetails:\n{json.dumps(self.details, indent=2)}"
        super().__init__(self.message)

    def _bytes_to_text(self, value: bytes) -> str:
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")

    def _extract_details(self, error: UnexpectedModelBehavior) -> dict[str, object]:
        details: dict[str, object] = {
            "unexpected_model_behavior_message": error.message,
            "unexpected_model_behavior_body": error.body,
        }
        cause = error.__cause__
        if isinstance(cause, openai.APIStatusError):
            request = cause.response.request
            request_body: str | None = None
            if request.content:
                request_body = self._bytes_to_text(request.content)
            response_text: str | None = None
            if cause.response.content:
                response_text = self._bytes_to_text(cause.response.content)
            details["request"] = {
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "body": request_body,
            }
            details["response"] = {
                "status_code": cause.response.status_code,
                "headers": dict(cause.response.headers),
                "body": response_text,
            }
            details["provider_error_body"] = cause.body
            return details
        if isinstance(cause, httpx.HTTPStatusError):
            request_body = None
            if cause.request.content:
                request_body = self._bytes_to_text(cause.request.content)
            response_text = None
            if cause.response.content:
                response_text = self._bytes_to_text(cause.response.content)
            details["request"] = {
                "method": cause.request.method,
                "url": str(cause.request.url),
                "headers": dict(cause.request.headers),
                "body": request_body,
            }
            details["response"] = {
                "status_code": cause.response.status_code,
                "headers": dict(cause.response.headers),
                "body": response_text,
            }
            return details
        if cause is not None:
            details["cause"] = {"type": cause.__class__.__name__, "message": str(cause)}
        request_from_context = self._request_from_context()
        if request_from_context is not None:
            details["request"] = request_from_context
            request_source_obj = self._request_source_from_context()
            if request_source_obj is not None:
                details["request_source"] = request_source_obj
            response_from_context = self._response_from_context()
            if response_from_context is not None:
                details["response"] = response_from_context
        elif "request" not in details:
            reconstructed_request = self._reconstructed_request_from_context()
            if reconstructed_request is not None:
                details["request"] = reconstructed_request
                details["request_source"] = "reconstructed_from_quick_agent_context"
        return details

    def _request_from_context(self) -> dict[str, object] | None:
        context = self.request_context
        if not isinstance(context, dict):
            return None
        request_obj = context.get("request")
        if isinstance(request_obj, dict):
            return request_obj
        return None

    def _response_from_context(self) -> dict[str, object] | None:
        context = self.request_context
        if not isinstance(context, dict):
            return None
        response_obj = context.get("response")
        if isinstance(response_obj, dict):
            return response_obj
        return None

    def _request_source_from_context(self) -> str | None:
        context = self.request_context
        if not isinstance(context, dict):
            return None
        source_obj = context.get("request_source")
        if isinstance(source_obj, str):
            return source_obj
        return None

    def _reconstructed_request_from_context(self) -> dict[str, object] | None:
        context = self.request_context
        if not isinstance(context, dict):
            return None
        base_url_obj = context.get("base_url")
        model_name_obj = context.get("model_name")
        user_prompt_obj = context.get("user_prompt")
        system_prompt_obj = context.get("system_prompt")
        instructions_obj = context.get("instructions")
        model_settings_obj = context.get("model_settings")
        if not isinstance(base_url_obj, str) or not isinstance(model_name_obj, str) or not isinstance(user_prompt_obj, str):
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
        url = f"{base_url_obj.rstrip('/')}/chat/completions"
        return {
            "method": "POST",
            "url": url,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(body, ensure_ascii=False),
        }

    def __str__(self) -> str:
        return self.message

    def to_curl(self) -> str:
        request_obj = self.details.get("request")
        if not isinstance(request_obj, dict):
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
