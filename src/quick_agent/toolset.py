"""Toolset module - replaces pydantic_ai FunctionToolset."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import JsonValue, create_model


@dataclass
class FunctionSchema:
    """Replaces pydantic_ai function_schema."""

    json_schema: dict[str, JsonValue]
    takes_ctx: bool
    name: str = ""
    description: str = ""

    def to_openai_tool(self) -> dict[str, object]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.json_schema,
            },
        }


@dataclass
class Tool:
    """Minimal tool wrapper replacing pydantic_ai Tool."""

    function: Callable[..., Any]
    name: str
    description: str | None
    function_schema: FunctionSchema


def _generate_tool_schema(func: Callable[..., Any]) -> dict[str, JsonValue]:
    """Generate JSON schema from function signature using pydantic."""
    sig = inspect.signature(func)
    field_defs: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        if param.annotation == inspect.Parameter.empty:
            continue

        if param.default == inspect.Parameter.empty:
            field_defs[name] = param.annotation
        else:
            field_defs[name] = (param.annotation, param.default)

    if not field_defs:
        return {"type": "object", "properties": {}}

    model_name = f"{func.__name__}_Params"
    DynamicModel = create_model(model_name, **field_defs)
    return DynamicModel.model_json_schema()


class AgentToolset:
    """Replaces pydantic_ai FunctionToolset for tool storage and schema generation."""

    def __init__(self) -> None:
        self.tools: dict[str, Tool] = {}

    def add_function(
        self,
        func: Callable[..., Any],
        name: str,
        description: str | None,
    ) -> Tool:
        """Add a function as a tool."""
        json_schema = _generate_tool_schema(func)
        schema = FunctionSchema(
            json_schema=json_schema,
            takes_ctx=False,
            name=name,
            description=description or "",
        )
        tool = Tool(
            function=func,
            name=name,
            description=description,
            function_schema=schema,
        )
        self.tools[name] = tool
        return tool
