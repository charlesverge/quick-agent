---
name: "Harness Agent Memory"
model:
  provider: "openai-compatible"
  base_url: "http://localhost:11434/v1"
  model_name: "qwen.qwen3-next-80b-a3b"
  temperature: 0.1
  max_completion_tokens: 256
tools:
  - personalize_results_tool
schemas:
  RandomWordOutput: "examples.agent_memory.schemas:RandomWordOutput"
chain:
  - id: generate_random_word
    kind: structured
    output_schema: RandomWordOutput
    prompt_section: step:generate
  - id: personalize
    kind: text
    prompt_section: step:personalize
output:
  format: text
---

## step:generate

Generate one random english word.
Return only JSON matching schema `RandomWordOutput` with key `random_word`.

## step:personalize

Read the previous step result and extract `random_word`.
Call tool `personalize_results_tool` with that value.
Return only the tool result text.
