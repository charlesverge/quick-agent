# Single-Shot and Structured Output

In this codebase, **true single-shot** means `chain` is omitted (or `chain: []`).
You can return structured output in this mode via `output.output_schema`.

Rules:

- `output.output_schema` requires an empty `chain`.
- `output.output_schema` must reference a schema key from `schemas`.
- Structured single-shot uses a direct OpenAI Chat Completions JSON-schema call (no tool call required).
- Because no tool call is used, `tools` are not supported with `output.output_schema`.
- `single_shot_use_pydantic_ai: true` switches structured single-shot back to the original pydantic-ai path.

## When To Use

Use true single-shot structured mode when you want:

- one model call
- deterministic output shape
- no intermediate draft/planning steps

## Structured Template (True Single-Shot)

Create `agents/<agent_id>.md` with no chain steps:

```markdown
---
name: "Single Shot Structured Agent"
description: "One-pass structured extraction"
model:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "gpt-5.2"
  temperature: 0
schemas:
  Output: "quick_agent.schemas.outputs:SummaryOutput"
# Optional: preserve original pydantic-ai single-shot structured path
# single_shot_use_pydantic_ai: true
output:
  format: json
  file: out/result.json
  output_schema: Output
---

## Instructions

You produce only data required by the output schema.
If a field cannot be inferred from input, return an empty string or empty list.

Read the input and return JSON that matches `Output`.
Do not include markdown, prose, or fields not in the schema.
```

## Run

```bash
quick-agent --agent <agent_id> --input safe/path/to/input.txt
```

If `output.file` is set, the parsed structured output is written as JSON.
If `output.file` is omitted, output is returned inline only.

## Structured Inline-Only Template

Use this when you want a structured object return value and no output file:

```markdown
---
name: "Single Shot Structured Inline"
model:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "gpt-5.2"
schemas:
  Output: "quick_agent.schemas.outputs:SummaryOutput"
output:
  format: json
  output_schema: Output
---

## Instructions

Return JSON only for the `Output` schema.
```

## Schema Template

Create or extend your schema module:

```python
from pydantic import BaseModel


class SummaryOutput(BaseModel):
    title: str
    bullets: list[str]
```

## Single-Shot Text Template

Use this when you want no chain and plain text output:

```markdown
---
name: "Single Shot Text Agent"
description: "One-pass text response"
model:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "gpt-5.2"
output:
  format: markdown
  file: out/result.md
---

## Instructions

Answer directly from the input in one response.
```

## Minimal Example

```markdown
---
name: "Ticket Summarizer"
model:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "gpt-5.2"
schemas:
  Output: "quick_agent.schemas.outputs:SummaryOutput"
output:
  format: json
  file: out/ticket_summary.json
  output_schema: Output
---

## Instructions

Summarize the input into:
- `title`: one short line
- `bullets`: 2 to 4 concise bullet strings
Return valid JSON for the `Output` schema only.
```
