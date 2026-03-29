---
name: "Chunk Items Summary Agent"
description: "Summarize chunk item outputs into one final summary."
model:
  provider: openai-compatible
  base_url: "http://localhost:11434/v1"
  model_name: qwen3:0.6b
  max_completion_tokens: 1024
chain:
  - id: summarize_items
    kind: text
    prompt_section: step:summarize_items
output:
  format: text
  file: out/large_context_chunk_items_summary.txt
handoff:
  enabled: false
  agent_id: null
  input_mode: "last_step_output_json"
---

## instructions

The input is JSON from a parent agent and contains an `items` list with per-chunk outputs.

## step:summarize\_items

Create one consolidated summary across all chunk items.

Requirements:

- capture the main topic
- merge non-overlapping key points
- remove duplicate details
- include major constraints or caveats

Return plain text only.
