"""Agent execution logic."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Type, TypeAlias, TypedDict

from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import FileInput, InputAdaptor, TextInput
from quick_agent.io_utils import write_output
from quick_agent.json_utils import extract_first_json_object
from quick_agent.models.loaded_agent_file import LoadedAgentFile
from quick_agent.models.chain_step_spec import ChainStepSpec
from quick_agent.models.model_spec import ModelSpec
from quick_agent.models.run_input import RunInput
from quick_agent.prompting import make_user_prompt
from quick_agent.tools_loader import import_symbol

StepOutput: TypeAlias = str | dict[str, Any]


class ChainState(TypedDict):
    agent_id: str
    steps: dict[str, StepOutput]
    final_output: StepOutput | None


class QuickAgent:
    def __init__(
        self,
        *,
        registry: AgentRegistry,
        tools: AgentTools,
        directory_permissions: DirectoryPermissions,
        agent_id: str,
        input_data: InputAdaptor | Path,
        extra_tools: list[str] | None,
        write_output: bool = True,
    ) -> None:
        self._registry: AgentRegistry = registry
        self._tools: AgentTools = tools
        self._directory_permissions: DirectoryPermissions = directory_permissions
        self._agent_id: str = agent_id
        self._input_data: InputAdaptor | Path = input_data
        self._extra_tools: list[str] | None = extra_tools
        self.loaded: LoadedAgentFile = self._registry.get(self._agent_id)
        output_file = self.loaded.spec.output.file
        self._write_output_file: bool = write_output and bool(output_file)
        safe_dir = self.loaded.spec.safe_dir
        if safe_dir is not None and Path(safe_dir).is_absolute():
            raise ValueError("safe_dir must be a relative path.")
        self.permissions: DirectoryPermissions = self._directory_permissions.scoped(safe_dir)
        if isinstance(self._input_data, InputAdaptor):
            input_adaptor = self._input_data
        else:
            input_adaptor = FileInput(self._input_data, self.permissions)
        self.run_input: RunInput = input_adaptor.load()

        self.tool_ids: list[str] = self._build_tool_ids()
        self.toolset: FunctionToolset[Any] | None = self._build_toolset()

        self.model: OpenAIChatModel = build_model(self.loaded.spec.model)
        self.model_settings_json: ModelSettings | None = self._build_model_settings(self.loaded.spec.model)
        self.state: ChainState = self._init_state()

    async def run(self) -> BaseModel | str:
        if self.has_tools():
            if self.toolset is None:
                raise ValueError("Toolset is missing while tools are enabled.")
            self._tools.maybe_inject_agent_call(
                self.tool_ids,
                self.toolset,
                self.run_input.source_path,
                self._run_nested_agent,
            )

        final_output = await self._run_chain()

        if self._write_output_file:
            self._write_final_output(final_output)

        await self._handle_handoff(final_output)

        return final_output

    async def _run_nested_agent(self, agent_id: str, input_data: InputAdaptor | Path) -> BaseModel | str:
        nested_write_output = self.loaded.spec.nested_output == "file"
        agent = QuickAgent(
            registry=self._registry,
            tools=self._tools,
            directory_permissions=self._directory_permissions,
            agent_id=agent_id,
            input_data=input_data,
            extra_tools=None,
            write_output=nested_write_output,
        )
        return await agent.run()

    def _init_state(self) -> ChainState:
        return {
            "agent_id": self._agent_id,
            "steps": {},
            "final_output": None,
        }

    def _build_model_settings(self, model_spec: ModelSpec) -> ModelSettings | None:
        if model_spec.provider == "openai-compatible":
            # Ollama OpenAI-compatible API uses "format": "json" to force JSON output.
            if model_spec.base_url != "https://api.openai.com/v1":
                return {"extra_body": {"format": "json"}}
        return None

    def _build_structured_model_settings(self, *, schema_cls: Type[BaseModel]) -> ModelSettings | None:
        model_settings: ModelSettings | None = self.model_settings_json
        provider = getattr(self.model, "provider", None)
        base_url = getattr(provider, "base_url", None)
        if base_url == "https://api.openai.com/v1":
            if self.model_settings_json is None:
                model_settings_dict: ModelSettings = {}
            else:
                model_settings_dict = self.model_settings_json
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
        step: ChainStepSpec,
    ) -> tuple[StepOutput, BaseModel | str]:
        if step.kind == "text":
            return await self._run_text_step(
                step=step,
            )

        if step.kind == "structured":
            return await self._run_structured_step(
                step=step,
            )

        raise NotImplementedError(f"Unknown step kind: {step.kind}")

    def _build_user_prompt(self) -> str:
        return make_user_prompt(self.run_input, self.state)

    def _build_step_instructions(self, step_prompt: str) -> str:
        if not self.loaded.instructions:
            return f"## Step Instructions\n{step_prompt}"
        return f"{self.loaded.instructions}\n\n## Step Instructions\n{step_prompt}"

    def _build_single_shot_prompt(self) -> str:
        return make_user_prompt(self.run_input, self.state)

    def _normalize_agent_text(self, text: str) -> str | None:
        if text:
            return text
        return None

    def _normalize_system_prompt(self, text: str) -> str | list[str]:
        if text:
            return text
        return []

    async def _run_text_step(
        self,
        *,
        step: ChainStepSpec,
    ) -> tuple[StepOutput, BaseModel | str]:
        user_prompt = self._build_user_prompt()
        step_prompt = self.loaded.step_prompts[step.prompt_section]
        step_instructions = self._build_step_instructions(step_prompt)
        toolsets = self._toolsets_for_run()
        agent = Agent(
            self.model,
            instructions=step_instructions,
            system_prompt=self._normalize_system_prompt(self.loaded.system_prompt),
            toolsets=toolsets,
            output_type=str,
        )
        result = await agent.run(user_prompt)
        return result.output, result.output

    async def _run_single_shot(self) -> BaseModel | str:
        user_prompt = self._build_single_shot_prompt()
        toolsets = self._toolsets_for_run()
        agent = Agent(
            self.model,
            instructions=self._normalize_agent_text(self.loaded.instructions),
            system_prompt=self._normalize_system_prompt(self.loaded.system_prompt),
            toolsets=toolsets,
            output_type=str,
        )
        result = await agent.run(user_prompt)
        return result.output

    async def _run_structured_step(
        self,
        *,
        step: ChainStepSpec,
    ) -> tuple[StepOutput, BaseModel | str]:
        if not step.output_schema:
            raise ValueError(f"Step {step.id} is structured but missing output_schema.")
        schema_cls = resolve_schema(self.loaded, step.output_schema)

        model_settings = self._build_structured_model_settings(schema_cls=schema_cls)

        user_prompt = self._build_user_prompt()
        step_prompt = self.loaded.step_prompts[step.prompt_section]
        step_instructions = self._build_step_instructions(step_prompt)
        toolsets = self._toolsets_for_run()
        agent = Agent(
            self.model,
            instructions=step_instructions,
            system_prompt=self._normalize_system_prompt(self.loaded.system_prompt),
            toolsets=toolsets,
            output_type=schema_cls,
            model_settings=model_settings,
        )
        result = await agent.run(user_prompt)
        raw_output = result.output
        if isinstance(raw_output, BaseModel):
            parsed = raw_output
        elif isinstance(raw_output, dict):
            parsed = schema_cls.model_validate(raw_output)
        else:
            try:
                parsed = schema_cls.model_validate_json(raw_output)
            except ValidationError:
                extracted = extract_first_json_object(raw_output)
                parsed = schema_cls.model_validate_json(extracted)
        return parsed.model_dump(), parsed

    async def _run_chain(
        self,
    ) -> BaseModel | str:
        if not self.loaded.spec.chain:
            return await self._run_single_shot()
        final_output: BaseModel | str = ""
        for step in self.loaded.spec.chain:
            step_out, step_final = await self._run_step(
                step=step,
            )
            self.state["steps"][step.id] = step_out
            self.state["final_output"] = step_out
            final_output = step_final
        return final_output

    def has_tools(self) -> bool:
        if not self.tool_ids:
            return False
        return True

    def _build_tool_ids(self) -> list[str]:
        if not self.loaded.spec.tools:
            return []
        return list(dict.fromkeys((self.loaded.spec.tools or []) + (self._extra_tools or [])))

    def _build_toolset(self) -> FunctionToolset[Any] | None:
        if not self.has_tools():
            return None
        return self._tools.build_toolset(self.tool_ids, self.permissions)

    def _toolsets_for_run(self) -> list[FunctionToolset[Any]]:
        if not self.has_tools():
            return []
        toolset = self.toolset
        if toolset is None:
            return []
        return [toolset]

    def _write_final_output(self, final_output: BaseModel | str) -> Path:
        output_file = self.loaded.spec.output.file
        if not output_file:
            raise ValueError("Output file is not configured.")
        out_path = Path(output_file)
        if isinstance(final_output, BaseModel):
            if self.loaded.spec.output.format == "json":
                write_output(out_path, final_output.model_dump_json(indent=2), self.permissions)
            else:
                write_output(out_path, final_output.model_dump_json(indent=2), self.permissions)
        else:
            write_output(out_path, str(final_output), self.permissions)
        return out_path

    async def _handle_handoff(self, final_output: BaseModel | str) -> None:
        if self.loaded.spec.handoff.enabled and self.loaded.spec.handoff.agent_id:
            if isinstance(final_output, BaseModel):
                payload = final_output.model_dump_json(indent=2)
            else:
                payload = str(final_output)
            await self._run_nested_agent(self.loaded.spec.handoff.agent_id, TextInput(payload))


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
