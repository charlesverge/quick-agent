"""Recorder helper for QuickAgent request and HTTP traffic logging."""

from __future__ import annotations

import json
import logging
import shlex
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic_ai.settings import ModelSettings

from quick_agent.json_utils import json_compatible_value
from quick_agent.models.model_spec import ModelSpec

logger = logging.getLogger(__name__)


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


class Recorder:
    def __init__(
        self,
        *,
        agent_id: str,
        model_spec: ModelSpec,
        effective_base_url: str,
        tool_ids: list[str],
        http_log_max_entries: int = 200,
        enable_llm_request_logging: bool = False,
        llm_log_path: Path | None = None,
    ) -> None:
        self._agent_id = agent_id
        self.model_spec = model_spec
        self.effective_base_url = effective_base_url
        self.tool_ids = tool_ids
        self._http_log_max_entries = http_log_max_entries
        self._http_traffic_entries: list[dict[str, object]] = []
        self._enable_llm_request_logging = enable_llm_request_logging
        self._llm_log_path = llm_log_path or Path("log/results.log")
        self.http_request_log: list[dict[str, object]] = []
        self.http_response_log: list[dict[str, object]] = []
        self.execution_log: list[ExecutionLogEntry] = []

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
                "base_url": self.effective_base_url,
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
            "model_settings": json_compatible_value(model_settings),
            "tool_ids": list(self.tool_ids),
        }
        self._write_llm_request_log(payload)

    def _unexpected_model_behavior_request_context(
        self,
        *,
        instructions: str | None,
        system_prompt: str | list[str],
        user_prompt: str,
        model_settings: ModelSettings | None,
    ) -> dict[str, object]:
        context: dict[str, object] = {
            "base_url": self.effective_base_url,
            "model_name": self.model_spec.model_name,
            "instructions": instructions,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "model_settings": json_compatible_value(model_settings),
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

    def _write_llm_request_log(self, payload: dict[str, object] | None) -> None:
        prefix = "Recorder._write_llm_request_log"
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
