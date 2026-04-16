---
name: "single_shot"
description: "Simple single-shot agent for batch test mode testing"
model:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "gpt-4o"
  temperature: 0.1
  max_completion_tokens: 2048
chain:
  - id: "step1"
    kind: "text"
    prompt_section: "step:step1"
output:
  format: "text"
---

## step:step1

Return the word: ok