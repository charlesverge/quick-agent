---
name: "chain"
description: "Chain agent for batch test mode testing"
model:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "gpt-4o"
  temperature: 0.1
  max_completion_tokens: 2048
tools:
  - "filesystem_append_text"
chain:
  - id: "step1"
    kind: "text"
    prompt_section: "step:step1"
  - id: "step2"
    kind: "text"
    prompt_section: "step:step2"
output:
  format: "text"
---

## step:step1

Append "step1 executed" to the file at {output_file}.

## step:step2

Append "step2 executed" to the file at {output_file}.