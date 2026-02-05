---
# Agent identity
name: "doc_pipeline_agent"
description: "Parse input, plan, execute tools, and emit validated structured output."

# Model configuration: OpenAI-compatible endpoint (e.g., SGLang server)
model:
  provider: "openai-compatible"
  base_url: "http://localhost:30000/v1"
  api_key_env: "SGLANG_API_KEY"
  model_name: "qwen/qwen2.5-32b-instruct"
  temperature: 0.2
  max_tokens: 2048

# Tools available to this agent (tool IDs resolved by the orchestrator)
tools:
  - "filesystem.read_text"
  - "filesystem.write_text"
  - "utils.json_merge"
  - "agent.call"              # orchestrator-provided tool for inter-agent calls

# Schema registry: symbolic names -> import paths
schemas:
  Plan: "schemas.outputs:Plan"
  FinalResult: "schemas.outputs:FinalResult"

# Prompt-chaining steps (ordered). Each step references a markdown section below.
chain:
  - id: "plan"
    kind: "structured"
    output_schema: "Plan"
    prompt_section: "step:plan"

  - id: "execute"
    kind: "text"
    prompt_section: "step:execute"

  - id: "finalize"
    kind: "structured"
    output_schema: "FinalResult"
    prompt_section: "step:finalize"

# Output settings
output:
  format: "json"
  file: "out/result.json"

# Optional: handoff to another agent after producing final output
handoff:
  enabled: false
  agent_id: "postprocess_agent"
  input_mode: "final_output_json"   # or "final_output_markdown"
---

# System Prompt

This is a system prompt to included in every run

## Instructions

Instructions are only included in first run.

You are a reliable pipeline agent.
You must follow the chain steps in order.
You may call tools as needed. If you call `agent.call`, wait for the response and then continue.

## step:plan

Goal:

- Read the provided input (a JSON or Markdown/text file) embedded by the orchestrator.
- Produce a structured **Plan** that lists concrete actions and any tool calls required.

Constraints:

- Keep steps explicit.
- If you need another agent, call `agent.call` with a clear request.

## step:execute

Goal:

- Execute the plan.
- Use the declared tools. You may call tools multiple times.

Constraints:

- Write intermediate artifacts only if asked.
- Summarize what you did in plain text.

## step:finalize

Goal:

- Produce a final **FinalResult** object that is valid JSON for the schema.
- Include references to tools invoked and any sub-agent calls.
- If anything failed, reflect it in the structured fields rather than “hiding” it in prose.

Return only the structured object required by the schema (no additional commentary).
