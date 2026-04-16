# Tools

Tools are Python functions that agents can call during execution. Each tool has a `tool.json`
registration file and a Python implementation.

## Tool Discovery

`AgentTools` accepts a list of tool root directories. At runtime it recursively scans each root
for `tool.json` files and builds a name-indexed registry. When an agent declares a tool by name,
the registry locates the matching entry and loads the implementation.

```python
from pathlib import Path
from quick_agent.agent_tools import AgentTools

tools = AgentTools([Path("src/quick_agent/tools"), Path("examples/my_feature")])
```

Built-in tools (`filesystem_*`, `shell_run`) are handled directly by adapters inside
`tools_loader.py`. All other tools load their implementation via `importlib.import_module`
using the `module` field in `tool.json`.

## tool.json Format

Every tool directory must contain a `tool.json` file:

```
<tool_root>/
  <tool_name>/
    tool.json
```

```json
{
  "name": "my_tool",
  "description": "What this tool does.",
  "impl": {
    "kind": "python",
    "module": "my_package.my_tool",
    "function": "my_tool"
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `name` | yes | Tool identifier used in agent frontmatter. Must not contain `.`. |
| `description` | yes | Shown to the model as the tool description. |
| `impl.kind` | yes | Always `"python"` currently. |
| `impl.module` | yes | Fully-qualified Python module path. Must be importable at runtime. |
| `impl.function` | yes | Function name within the module. |

The input parameter schema is generated automatically from the Python function's
type hints at load time. Both interactive and batch modes use the same introspection
path, so there is no need to declare the schema manually in `tool.json`.

## Python Implementation

The implementation function signature depends on whether the tool needs agent context (memory,
deps) or is a plain function.

### Plain function

No context needed — accepts only the declared input parameters:

```python
def my_tool(value: str) -> str:
    return f"processed: {value}"
```

### Context-aware function

Use `RunContext` from `pydantic_ai` to access agent memory and deps:

```python
from pydantic_ai import RunContext


def my_tool(ctx: RunContext[dict[str, object]], value: str) -> str:
    memory = ctx.deps["memory"]
    name = memory["first_name"]
    return f"{name}: {value}"
```

The `ctx.deps` dict contains `"memory"` — the dict passed as `memory=` when constructing
`QuickAgent` or calling `Orchestrator.batch(..., memory={...})`.

## Module Importability

The `module` path in `tool.json` is resolved with `importlib.import_module`. The package root
must be on `sys.path` at the time the tool is loaded.

- Tools under `src/quick_agent/tools/` use module paths like `quick_agent.tools.filesystem.append_text`.
  These are importable because `src/` is on `sys.path` via the editable install.

- Tools in project sub-directories (e.g. `examples/my_feature/`) use paths like
  `examples.my_feature.my_tool`. These require the **repo root** to be on `sys.path`, which
  means the script must be run from the repo root:

  ```bash
  # correct — repo root is added to sys.path
  python deploy/my_harness/run.py

  # incorrect — harness dir is added instead of repo root
  cd deploy/my_harness && ./run.py
  ```

## Registering a Tool in an Agent

Declare tool names in the agent frontmatter `tools` list:

```markdown
---
name: "My Agent"
tools:
  - my_tool
---
```

The name must match the `name` field in `tool.json` exactly.

## Per-Agent Tool Directories

Different agents can use different tool sets by constructing separate `AgentTools` instances
and passing them directly to `QuickAgent`:

```python
from quick_agent.agent_tools import AgentTools
from quick_agent.quick_agent import QuickAgent

memory_tools = AgentTools([Path("examples/agent_memory")])

agent = QuickAgent(
    registry=orchestrator.registry,
    tools=memory_tools,
    directory_permissions=orchestrator.directory_permissions,
    agent_id="my-memory-agent",
    input_data=TextInput("..."),
    extra_tools=None,
    memory={"first_name": "Charles"},
)
request = agent.batch()
```

This pattern is used in the bedrock batch test harness (`setup.py`) to give the agent-memory
agent its own tool root (`examples/agent_memory`) without exposing filesystem tools to it.

## Built-in Tools

| Name | Description |
|------|-------------|
| `filesystem_read_text` | Read a file within the permitted directory |
| `filesystem_write_text` | Write a file within the permitted directory |
| `filesystem_append_text` | Append to a file within the permitted directory |
| `filesystem_list_files` | List files in a directory |
| `filesystem_find_closest_file` | Find the closest filename match in a directory |
| `filesystem_delete_file` | Delete a file within the permitted directory |
| `shell_run` | Run a shell command within the permitted directory |

Built-in tools are wired to adapter classes (`FilesystemToolAdapter`, `ShellToolAdapter`) and do
not require a `module` import — they are matched by name in `tools_loader.py`.

## Tool Choice

`tool_choice` controls how models call tools and can be configured at:

- agent level (`AgentSpec.tool_choice`)
- chain step level (`ChainStepSpec.tool_choice`)

If both are set, chain step value takes precedence.

### Modes

- `auto`: default model behavior
- `required`: model must call at least one tool
- `none`: disable tool calling for the request
- `any`: Bedrock-only forcing mode; non-Bedrock providers resolve this to `auto`

`tool_choice` can be provided as shorthand string or object:

```yaml
tool_choice: "required"
```

```yaml
tool_choice:
  type: "function"
  name: "filesystem_list_files"
```

### Allowed Tools

You can constrain the outbound tool list:

```yaml
tool_choice:
  allowed_tools:
    - name: "filesystem_list_files"
```

This is payload-only enforcement: only listed tools are sent to the model for that request.
