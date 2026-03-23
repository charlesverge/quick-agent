---
name: "Large Context Sample Then Chunk Agent"
description: "Sample first, then paragraph chunking, then summarize each chunk."
content_processing:
  sample:
    ratios: [25, 50, 25]
    max_chunk_tokens: 4000
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
  file: out/large_context_sample_then_chunk.json
---

## instructions

Summarize each provided input chunk concisely.

## step:summarize

Return plain text summary with:

- main point
- key details
- constraints
