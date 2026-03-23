---
name: "Sample 25-50-25 Summary Agent"
description: "Summarize a large document after 25/50/25 sampling to a 2000-token budget."
content_processing:
  sample:
    ratios: [25, 50, 25]
    max_chunk_tokens: 2000
    debug_output_file: test.log
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
  file: out/sample_25_50_25_summary.json
---

## instructions

Create a concise factual summary from the provided content.

## step:summarize

Produce a summary with:

- main purpose
- key points
- notable constraints or caveats

Return plain text only.
