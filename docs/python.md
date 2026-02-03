# Using As A Python Module

You can embed the orchestrator in your own scripts.

## Minimal Example (User Agents/Tools Only)

```python
import anyio
from pathlib import Path

from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.orchestrator import Orchestrator

agent_roots = [Path("agents")]
tool_roots = [Path("tools")]
permissions = DirectoryPermissions(Path("safe"))

orchestrator = Orchestrator(agent_roots, tool_roots, directory_permissions=permissions)

async def main() -> None:
    result = await orchestrator.run_agent("example", Path("input.txt"))
    print(result)

anyio.run(main)
```

## Inter-Agent Calls

Use the `agent.call` tool id in your agent front matter, then invoke the tool by its function name `agent_call` from the prompt.

Example agent pair:

`agents/child.md`:

```markdown
---
name: "Child Agent"
chain:
  - id: respond
    kind: text
    prompt_section: step:respond
output:
  format: json
  file: out/child.json
---

## step:respond

Reply with exactly: pong
```

`agents/parent.md`:

```markdown
---
name: "Parent Agent"
tools:
  - "agent.call"
chain:
  - id: invoke
    kind: text
    prompt_section: step:invoke
output:
  format: json
  file: out/parent.json
---

## step:invoke

Call agent_call with agent "child" and input_file "{base_directory}/child_input.txt".
Then respond with only the returned text value.
```

`input/child_input.txt`:

```
ignored
```

Then run:

```bash
python agent.py --agent parent --input input/parent_input.txt
```

Expected output: `pong`

## Including Packaged System Agents And Tools

When installed as a module, packaged system assets live inside the
`quick_agent` package. Use `importlib.resources` to locate them:

```python
import anyio
from importlib.resources import files
from pathlib import Path

from quick_agent.directory_permissions import DirectoryPermissions
from quick_agent.orchestrator import Orchestrator

package_root = Path(files("quick_agent"))

agent_roots = [Path("agents"), package_root / "agents"]
tool_roots = [Path("tools"), package_root / "tools"]
permissions = DirectoryPermissions(Path("safe"))

orchestrator = Orchestrator(agent_roots, tool_roots, directory_permissions=permissions)

async def main() -> None:
    result = await orchestrator.run_agent("example", Path("input.txt"))
    print(result)

anyio.run(main)
```
