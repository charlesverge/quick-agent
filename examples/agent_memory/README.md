# Agent Memory Example

This example has two steps:

1.  A structured step generates a random word.

1.  A text step calls `personalize_results_tool`, which reads `first_name` from tool memory and returns:
   `"{first_name} your random word is {random_word}"`

## Files

- `agent_memory.md` agent definition
- `schemas.py` structured output schema for step one
- `personalize_results_tool.py` tool implementation using `RunContext.deps`
- `personalize_results_tool/tool.json` tool registration

## Run From Python (with memory)

```python
import anyio
from pathlib import Path

from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.input_adaptors import TextInput
from quick_agent.quick_agent import QuickAgent


async def main() -> None:
  registry = AgentRegistry([Path("examples/agent_memory")])
  tools = AgentTools([Path("examples/agent_memory")])
  permissions = DirectoryPermissions(Path("."))

  agent = QuickAgent(
    registry=registry,
    tools=tools,
    directory_permissions=permissions,
    agent_id="agent_memory",
    input_data=TextInput("Generate and personalize a random word."),
    extra_tools=None,
    memory={"first_name": "Charles"},
  )

  result = await agent.run()
  print(result)


anyio.run(main)
```

Expected text output pattern:

```text
Charles your random word is <random_word>
```
