# Agent Context And Memory

`QuickAgent` passes a dependency object to tools through `RunContext.deps`.
The dependency object contains:

- `state`: current chain state for the running agent
- `memory`: mutable memory dictionary owned by the running agent

Memory is for tools. Prompt steps do not read `memory` directly.

## Passing Memory At Initialization

```python
from pathlib import Path

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import TextInput
from quick_agent.quick_agent import QuickAgent

registry = AgentRegistry([Path("examples/agent_memory")])
tools = AgentTools([Path("examples/agent_memory")])
permissions = DirectoryPermissions(Path("."))

agent = QuickAgent(
  registry=registry,
  tools=tools,
  directory_permissions=permissions,
  agent_id="agent_memory",
  input_data=TextInput("Create one random word and personalize it."),
  extra_tools=None,
  memory={"first_name": "Charles"},
)
```

## Reading And Updating Memory After Init

```python
current = agent.memory
agent.memory = {"first_name": "Charles", "team": "alpha"}
```

## Reading Memory Inside A Tool

```python
from pydantic_ai import RunContext


def personalize_results_tool(ctx: RunContext[dict[str, object]], random_word: str) -> str:
  memory_obj = ctx.deps["memory"]
  if not isinstance(memory_obj, dict):
    raise TypeError("memory must be a dict.")
  first_name_obj = memory_obj["first_name"]
  if not isinstance(first_name_obj, str):
    raise TypeError("memory.first_name must be a string.")
  return f"{first_name_obj} your random word is {random_word}"
```

## Example

See [examples/agent_memory/readme.md](/Users/devuser/dev/personal/quick-agent/examples/agent_memory/readme.md) for a full two-step example using `personalize_results_tool`.
