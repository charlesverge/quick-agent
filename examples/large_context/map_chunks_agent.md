---
name: "Large Context Map Chunks Agent"
description: "Chunk large input using map_chunks and summarize each chunk."
content_processing:
  chunk_processing:
    mode: map_chunks
    provider: semchunks
    max_chunk_tokens: 1200
    overlap_percent: 10
model:
  provider: openai-compatible
  base_url: "http://localhost:11434/v1"
  model_name: qwen3:0.6b
  max_completion_tokens: 1024
chain:
  - id: summarize
    kind: text
    prompt_section: step:summarize
output:
  format: json
  file: out/large_context_map_chunks.json
---

## instructions

Summarize each provided input chunk concisely.

## step:summarize

Return plain text summary with:

- main point
- key details
- constraints
