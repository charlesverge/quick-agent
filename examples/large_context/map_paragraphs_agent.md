---
name: "Large Context Map Paragraphs Agent"
description: "Chunk large input by paragraph boundaries and summarize each chunk."
content_processing:
  chunk_processing:
    mode: map_paragraphs
    provider: semchunks
    max_chunk_tokens: 1200
    overlap_percent: 10
model:
  provider: openai-compatible
  base_url: "http://localhost:11434/v1"
  model_name: qwen3:0.6b
  max_tokens: 1024
chain:
  - id: summarize
    kind: text
    prompt_section: step:summarize
output:
  format: json
  file: out/large_context_map_paragraphs.json

handoff:
  enabled: true
  agent_id: "chunk_items_summary_agent"
  input_mode: "last_step_output_json"
nested_output: file
---

## instructions

Summarize each provided input chunk concisely.

## step:summarize

Return plain text summary with:

- main point
- key details
- constraints
