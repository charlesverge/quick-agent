# Outputs

This document describes how to control outputs in an `agent.md` file.

## Output Section

Use the `output:` section in front matter to configure the final output.

```yaml
output:
  format: json
  file: out/result.json
```

### format

- `json` (default): Write JSON when the final output is structured, otherwise write the raw text.
- `markdown`: Write the raw text output (structured outputs are still serialized as JSON).

### file

- If `output.file` is set, the top-level agent writes the final output to that path.
- If `output.file` is omitted, the top-level agent returns the final output inline and **does not write a file**.

## Structured Outputs

Structured steps use `kind: structured` with an `output_schema` mapped in `schemas:`.
If the **final** chain step is structured, the **final output** of the agent run is the parsed schema.
If `output.file` is configured, that parsed schema is written as JSON. If `output.file` is omitted, the parsed schema is returned inline only.

```yaml
schemas:
  Summary: "quick_agent.schemas.outputs:SummaryOutput"
chain:
  - id: summarize
    kind: structured
    prompt_section: step:summarize
    output_schema: Summary
```

## Nested Outputs

Nested agents invoked via `agent_call` or `handoff` do not write output files by default. Control this with `nested_output` in the parent agent front matter:

### json output

```yaml
nested_output: inline  # default, no file for nested calls
nested_output: file  # allow nested agents to write output.file
```

## Last chain output vs compiled output

By default the last chain is the return from a run. You can also configure the agent to return the compiled output.

Format json will return the state, which contains a dict with a field name for each step

```yaml
output:
  format: json
  file: out/result.json
  return_compiled_output: true
```

Example compiled output:

```json
{
  "step1": {"field1": "value1"},
  "step2": {"field2": "value2"},
  "last_step_output": {"field2": "value2"}
}
```

### Text output

Text output will concatenate the text output of each step in the chain, separated by newlines.

```yaml
output:
  file: out/result.txt
  return_compiled_output: true
```

Example compiled output:

```text
step1 result 1
step2 result 2
```

### Structured output

For structured output, a schema must be defined and for each step which you want an output a field name must exist.

```yaml
output:
  format: structured
  schema: "schemas.outputs:FinalOutput"
  return_compiled_output: true
```

```python
class FinalOutput(BaseModel):
    step1: Step1Output
    step2: Step2Output
    step3: Step3Output
    last_step_output: step2: Step3Output
```

```python
class FinalOutput(BaseModel):
    step2: Step2Output
    step3: Step2Output
```

## Notes

- Only the top-level agent run writes output files (when `output.file` is set).
- If the orchestrator safe directory is not configured, file writes are denied regardless of `output.file`.
