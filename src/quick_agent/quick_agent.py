"""Agent execution logic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.io_utils import load_input, write_output
from quick_agent.json_utils import extract_first_json_object
from quick_agent.models.loaded_agent_file import LoadedAgentFile
from quick_agent.models.model_spec import ModelSpec
from quick_agent.prompting import make_user_prompt
from quick_agent.tools_loader import import_symbol


class QuickAgent:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        tools: AgentTools,
        directory_permissions: DirectoryPermissions,
        agent_id: str,
        input_path: Path,
        extra_tools: list[str] | None,
    ) -> None:
        self._registry = registry
        self._tools = tools
        self._directory_permissions = directory_permissions
        self._agent_id = agent_id
        self._input_path = input_path
        self._extra_tools = extra_tools

        self.loaded = self._registry.get(self._agent_id)
        safe_dir = self.loaded.spec.safe_dir
        if safe_dir is not None and Path(safe_dir).is_absolute():
            raise ValueError("safe_dir must be a relative path.")
        self.permissions = self._directory_permissions.scoped(safe_dir)
        self.run_input = load_input(self._input_path, self.permissions)

        self.tool_ids = list(dict.fromkeys((self.loaded.spec.tools or []) + (self._extra_tools or [])))
        self.toolset = self._tools.build_toolset(self.tool_ids, self.permissions)

        self.model = build_model(self.loaded.spec.model)
        self.model_settings_json = self._build_model_settings(self.loaded.spec.model)
        self.state = self._init_state(self._agent_id)

    async def run(self) -> BaseModel | str:
        self._tools.maybe_inject_agent_call(
            self.tool_ids,
            self.toolset,
            self.run_input.source_path,
            self._run_nested_agent,
        )

        final_output = await self._run_chain(
            loaded=self.loaded,
            model=self.model,
            model_settings_json=self.model_settings_json,
            toolset=self.toolset,
            run_input=self.run_input,
            state=self.state,
        )

        out_path = self._write_final_output(self.loaded, final_output, self.permissions)

        await self._handle_handoff(self.loaded, out_path)

        return final_output

    async def _run_nested_agent(self, agent_id: str, input_path: Path) -> BaseModel | str:
        agent = QuickAgent(
            registry=self._registry,
            tools=self._tools,
            directory_permissions=self._directory_permissions,
            agent_id=agent_id,
            input_path=input_path,
            extra_tools=None,
        )
        return await agent.run()

    def _init_state(self, agent_id: str) -> dict[str, Any]:
        return {
            "agent_id": agent_id,
            "steps": {},
        }

    def _build_model_settings(self, model_spec: ModelSpec) -> ModelSettings | None:
        if model_spec.provider == "openai-compatible":
            # Ollama OpenAI-compatible API uses "format": "json" to force JSON output.
            if model_spec.base_url != "https://api.openai.com/v1":
                return {"extra_body": {"format": "json"}}
        return None

    def _build_structured_model_settings(
        self,
        *,
        model: OpenAIChatModel,
        model_settings_json: ModelSettings | None,
        schema_cls: Type[BaseModel],
    ) -> ModelSettings | None:
        model_settings: ModelSettings | None = model_settings_json
        provider = getattr(model, "provider", None)
        base_url = getattr(provider, "base_url", None)
        if base_url == "https://api.openai.com/v1":
            if model_settings_json is None:
                model_settings_dict: ModelSettings = {}
            else:
                model_settings_dict = model_settings_json
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
        step: Any,
        loaded: LoadedAgentFile,
        model: OpenAIChatModel,
        model_settings_json: ModelSettings | None,
        toolset: FunctionToolset[Any],
        run_input: Any,
        state: dict[str, Any],
    ) -> tuple[Any, BaseModel | str]:
        if step.kind == "text":
            return await self._run_text_step(
                step=step,
                loaded=loaded,
                model=model,
                toolset=toolset,
                run_input=run_input,
                state=state,
            )

        if step.kind == "structured":
            return await self._run_structured_step(
                step=step,
                loaded=loaded,
                model=model,
                model_settings_json=model_settings_json,
                toolset=toolset,
                run_input=run_input,
                state=state,
            )

        raise NotImplementedError(f"Unknown step kind: {step.kind}")

    def _build_user_prompt(
        self,
        *,
        step: Any,
        loaded: LoadedAgentFile,
        run_input: Any,
        state: dict[str, Any],
    ) -> str:
        if step.prompt_section not in loaded.step_prompts:
            raise KeyError(f"Missing step section {step.prompt_section!r} in agent.md body.")

        step_prompt = loaded.step_prompts[step.prompt_section]
        return make_user_prompt(step_prompt, run_input, state)

    async def _run_text_step(
        self,
        *,
        step: Any,
        loaded: LoadedAgentFile,
        model: OpenAIChatModel,
        toolset: FunctionToolset[Any],
        run_input: Any,
        state: dict[str, Any],
    ) -> tuple[Any, BaseModel | str]:
        user_prompt = self._build_user_prompt(
            step=step,
            loaded=loaded,
            run_input=run_input,
            state=state,
        )
        agent = Agent(
            model,
            instructions=loaded.body,
            toolsets=[toolset],
            output_type=str,
        )
        result = await agent.run(user_prompt)
        return result.output, result.output

    async def _run_structured_step(
        self,
        *,
        step: Any,
        loaded: LoadedAgentFile,
        model: OpenAIChatModel,
        model_settings_json: ModelSettings | None,
        toolset: FunctionToolset[Any],
        run_input: Any,
        state: dict[str, Any],
    ) -> tuple[Any, BaseModel | str]:
        if not step.output_schema:
            raise ValueError(f"Step {step.id} is structured but missing output_schema.")
        schema_cls = resolve_schema(loaded, step.output_schema)

        model_settings = self._build_structured_model_settings(
            model=model,
            model_settings_json=model_settings_json,
            schema_cls=schema_cls,
        )

        user_prompt = self._build_user_prompt(
            step=step,
            loaded=loaded,
            run_input=run_input,
            state=state,
        )
        agent = Agent(
            model,
            instructions=loaded.body,
            toolsets=[toolset],
            output_type=str,
            model_settings=model_settings,
        )
        result = await agent.run(user_prompt)
        raw_output = result.output
        try:
            parsed = schema_cls.model_validate_json(raw_output)
        except ValidationError:
            extracted = extract_first_json_object(raw_output)
            parsed = schema_cls.model_validate_json(extracted)
        return parsed.model_dump(), parsed

    async def _run_chain(
        self,
        *,
        loaded: LoadedAgentFile,
        model: OpenAIChatModel,
        model_settings_json: ModelSettings | None,
        toolset: FunctionToolset[Any],
        run_input: Any,
        state: dict[str, Any],
    ) -> BaseModel | str:
        final_output: BaseModel | str = ""
        for step in loaded.spec.chain:
            step_out, step_final = await self._run_step(
                step=step,
                loaded=loaded,
                model=model,
                model_settings_json=model_settings_json,
                toolset=toolset,
                run_input=run_input,
                state=state,
            )
            state["steps"][step.id] = step_out
            final_output = step_final
        return final_output

    def _write_final_output(
        self,
        loaded: LoadedAgentFile,
        final_output: BaseModel | str,
        permissions: DirectoryPermissions,
    ) -> Path:
        out_path = Path(loaded.spec.output.file)
        if isinstance(final_output, BaseModel):
            if loaded.spec.output.format == "json":
                write_output(out_path, final_output.model_dump_json(indent=2), permissions)
            else:
                write_output(out_path, final_output.model_dump_json(indent=2), permissions)
        else:
            write_output(out_path, str(final_output), permissions)
        return out_path

    async def _handle_handoff(self, loaded: LoadedAgentFile, out_path: Path) -> None:
        if loaded.spec.handoff.enabled and loaded.spec.handoff.agent_id:
            # For a more robust version, generate an intermediate file for handoff input.
            await self._run_nested_agent(loaded.spec.handoff.agent_id, out_path)


def resolve_schema(loaded: LoadedAgentFile, schema_name: str) -> Type[BaseModel]:
    if schema_name not in loaded.spec.schemas:
        raise KeyError(f"Schema {schema_name!r} not registered in agent.md schemas.")
    cls = import_symbol(loaded.spec.schemas[schema_name])
    if not isinstance(cls, type) or not issubclass(cls, BaseModel):
        raise TypeError(f"Schema {schema_name!r} must be a Pydantic BaseModel subclass.")
    return cls


def build_model(model_spec: ModelSpec) -> OpenAIChatModel:
    api_key = os.environ.get(model_spec.api_key_env, "noop")
    provider = OpenAIProvider(base_url=model_spec.base_url, api_key=api_key)
    return OpenAIChatModel(model_spec.model_name, provider=provider)
