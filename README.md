# Simple Agent

Simple Agent is a minimal, local-first agent runner that loads agent definitions from Markdown front matter and executes a small chain of steps with limited context handling. It is intentionally small and explicit: you define the model, tools, and steps in a single Markdown file, and the orchestrator runs those steps in order with a bounded prompt preamble.

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
python agent.py --agent hello --input safe/path/to/input.txt
```

Note: by default, file access is restricted to the `safe/` directory (use `--safe-dir` to change it).
Agents can further restrict access with `safe_dir` in frontmatter (relative to the safe root).

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
python agent.py --agent structured --input safe/path/to/input.txt
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
python agent.py --agent openai --input safe/path/to/input.txt
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
