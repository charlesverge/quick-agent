# Quick Agent

Quick Agent is a minimal, local-first agent runner that loads agent definitions from Markdown front matter and executes a small chain of steps with limited context handling. It is intentionally small and explicit: you define the model, tools, and steps in a single Markdown file, and the orchestrator runs those steps in order with a bounded prompt preamble.

## Project Goal

Provide a simple, maintainable agent framework that:
- Uses Markdown front matter for agent configuration.
- Runs a deterministic chain of steps (text or structured output).
- Keeps context handling deliberately limited and predictable.
- Supports local tools and simple inter-agent calls.

## Install

```bash
pip install quick-agent
```

If you are working from source:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
```

## Hello World Example

Create `agents/hello.md`:

```markdown
---
name: "Hello Agent"
description: "Minimal example"
model:
  provider: "openai-compatible"
  base_url: "http://localhost:11434/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "llama3"
chain:
  - id: hello
    kind: text
    prompt_section: step:hello
output:
  format: json
  file: out/hello.json
---

## step:hello

Say hello to the input.
```

Then run:

```bash
quick-agent --agent hello --input safe/path/to/input.txt
```

Note: by default, file access is restricted to the `safe/` directory (use `--safe-dir` to change it).
Agents can further restrict access with `safe_dir` in frontmatter (relative to the safe root).

If you omit the entire `model:` section, the defaults are:

```yaml
model:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "gpt-5.2"
```

## Structured Output Example

Create `agents/structured.md`:

```markdown
---
name: "Structured Agent"
model:
  provider: "openai-compatible"
  base_url: "http://localhost:11434/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "llama3"
schemas:
  Summary: "quick_agent.schemas.outputs:SummaryOutput"
chain:
  - id: summarize
    kind: structured
    prompt_section: step:summarize
    output_schema: Summary
output:
  format: json
  file: out/summary.json
---

## step:summarize

Summarize the input into a short title and 2 bullet points.
```

Define the schema in `src/quick_agent/schemas/outputs.py` (example below):

```python
from pydantic import BaseModel


class SummaryOutput(BaseModel):
    title: str
    bullets: list[str]
```

Then run:

```bash
quick-agent --agent structured --input safe/path/to/input.txt
```

## OpenAI API Example

This example uses OpenAI's API via the OpenAI-compatible provider. Set your API key in the environment:

```bash
export OPENAI_API_KEY="your-key"
```

Create `agents/openai.md`:

```markdown
---
name: "OpenAI Agent"
chain:
  - id: reply
    kind: text
    prompt_section: step:reply
output:
  format: json
  file: out/openai.json
---

## step:reply

Answer the input in a short paragraph.
```

Then run:

```bash
quick-agent --agent openai --input safe/path/to/input.txt
```

## Python Usage

You can also run agents programmatically:

```python
from pathlib import Path

import anyio

from quick_agent import Orchestrator, QuickAgent
from quick_agent.agent_registry import AgentRegistry
from quick_agent.agent_tools import AgentTools
from quick_agent.directory_permissions import DirectoryPermissions


def main() -> None:
    agent_roots = [Path("agents")]
    tool_roots = [Path("tools")]
    safe_dir = Path("safe")

    registry = AgentRegistry(agent_roots)
    tools = AgentTools(tool_roots)
    permissions = DirectoryPermissions(safe_dir)

    agent = QuickAgent(
        registry=registry,
        tools=tools,
        directory_permissions=permissions,
        agent_id="hello",
        input_path=Path("safe/path/to/input.txt"),
        extra_tools=None,
    )

    result = anyio.run(agent.run)
    print(result)


if __name__ == "__main__":
    main()
```

## How It Works

Agents are stored as Markdown files with YAML front matter and step sections:

- Front matter declares the model, tools, schemas, and chain.
- Body contains `## step:<id>` sections referenced by the chain.

The orchestrator loads the agent, builds the tools, and executes each step in order, writing the final output to disk.

## Documentation

See the docs in `docs/`:

- [docs/cli.md](docs/cli.md): Command line usage and options.
- [docs/templates.md](docs/templates.md): Agent template format and examples.
- [docs/python.md](docs/python.md): Embedding the orchestrator in scripts.
- [docs/python.md#inter-agent-calls](docs/python.md#inter-agent-calls): Example of one agent calling another.
