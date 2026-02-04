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

```yaml
nested_output: inline  # default, no file for nested calls
# nested_output: file  # allow nested agents to write output.file
```

## Notes

- Only the top-level agent run writes output files (when `output.file` is set).
- If the orchestrator safe directory is not configured, file writes are denied regardless of `output.file`.
