---
name: "with_tools"
description: "Agent with tools for batch test mode testing"
model:
  provider: "openai-compatible"
  base_url: "https://api.openai.com/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "gpt-4o"
  temperature: 0.1
  max_completion_tokens: 2048
tools:
  - "filesystem_read_text"
  - "filesystem_append_text"
chain:
  - id: "execute"
    kind: "text"
    prompt_section: "step:execute"
output:
  format: "text"
---

## step:execute

1. Read the file at {notes_file}
2. Append " - appended by test" to the same file
3. Return "done"