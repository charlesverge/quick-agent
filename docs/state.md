# State Management Chain

This document summarizes how Quick Agent manages chain state during execution, based on the current implementation.

## State Shape

Quick Agent uses a simple dictionary state with two keys:

- `agent_id`: the id of the running agent.
- `steps`: a mapping of `step_id` to the stored output for each completed step.

This shape is created in `QuickAgent._init_state` and stored on the instance (`QuickAgent.state`).

## How State Flows Through Steps

Steps come from the agent’s `chain` spec and run in order.

1. Build the user prompt with `make_user_prompt` (input metadata/text, current state JSON, step prompt text).
2. Run the step as either `text` or `structured`.
3. Store the step output in `state["steps"][step.id]`.

After the last step, the final output is the return value of the last step.

## Step Output Types

- `text` steps store a string in `state["steps"][step.id]`.
- `structured` steps validate the model output against a Pydantic schema and store the parsed dict in `state["steps"][step.id]`. If the model response contains extra text, Quick Agent extracts the first JSON object and retries validation.

## Practical Example (text Output)

The agent file `agents/business-extract.md` defines a three-step chain that stores each extraction in `state.steps` and uses prior outputs in the final summary step.

Chain excerpt:

```markdown
chain:
  - id: company_name
    kind: text
    prompt_section: step:company_name
  - id: location
    kind: text
    prompt_section: step:location
  - id: summary
    kind: text
    prompt_section: step:summary
```

Step excerpt:

```markdown
## step:summary

Write one sentence summarizing the business.
Use `state.steps.company_name` and `state.steps.location` if available.
```

## Practical Example (Structured Output)

The agent file `agents/business-extract-structured.md` uses a structured output schema for the final step.

Chain excerpt:

```markdown
chain:
  - id: company_name
    kind: text
    prompt_section: step:company_name
  - id: location
    kind: text
    prompt_section: step:location
  - id: summary
    kind: structured
    prompt_section: step:summary
    output_schema: BusinessSummary
```

Schema excerpt:

```markdown
schemas:
  BusinessSummary: "quick_agent.schemas.outputs:BusinessSummary"
```

Summary step excerpt:

```markdown
## step:summary

Return a JSON object with:
- `company_name`
- `location`
- `summary` (one sentence)

Use `state.steps.company_name` and `state.steps.location` for the fields if available.
```

## Output and Handoff

After the chain completes:

- If output writing is enabled and `output.file` is set, the final output is written to the configured file.
- If handoff is enabled, the final output is passed to the next agent. Structured outputs are serialized as JSON before handoff.
