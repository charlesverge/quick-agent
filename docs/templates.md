# Templates

Agents are defined as Markdown files with YAML frontmatter and step sections.

## Agent File Layout

```markdown
---
name: "Example Agent"
description: "Short description"
model:
  provider: "openai-compatible"
  base_url: "http://localhost:11434/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "llama3"
  temperature: 0.2
  max_tokens: 2048
tools:
  - filesystem.read_text
  - filesystem.write_text
safe_dir: "examples/agent-a"
schemas:
  Output: "quick_agent.schemas.outputs:ExampleOutput"
chain:
  - id: draft
    kind: text
    prompt_section: step:draft
  - id: final
    kind: structured
    prompt_section: step:final
    output_schema: Output
output:
  format: json
  file: out/result.json
handoff:
  enabled: false
  agent_id: null
nested_output: inline
---

## step:draft

Write a first draft.

## step:final

Produce final structured output.
```

## Required Fields

- `name`
- `model.base_url`
- `model.model_name`
- `chain` (at least one step)

## Step Sections

Each step in `chain` must reference a matching `prompt_section` in the body:

```markdown
## step:<id>
```

## Tools

Tools are loaded by id from `tool.json` files.
A tool definition folder contains a `tool.json` with the `id` matching what you use in `tools`:

```json
{
  "id": "filesystem.read_text",
  "name": "filesystem.read_text",
  "description": "Read a UTF-8 file",
  "impl": {
    "kind": "python",
    "module": "quick_agent.tools.filesystem.read_text",
    "function": "read_text"
  }
}
```

## Safe Directory

Set `safe_dir` to restrict an agent to a subdirectory of the orchestrator safe root.
The value must be a relative path.

## Schemas

For structured steps, map schema names in `schemas` to import paths:

```yaml
schemas:
  Output: "your_package.schemas:OutputModel"
```

`output_schema` must reference one of those keys.

## Nested Output

Set `nested_output` to control whether agents invoked via `agent_call` or `handoff`
write their output file.

- `inline` (default): nested agents return output only; no file is written.
- `file`: nested agents write their configured `output.file`.
