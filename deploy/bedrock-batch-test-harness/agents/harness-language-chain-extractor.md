---
name: "Harness Language Extractor"
description: "Extract programming languages and technical skills from markdown input."
model:
  provider: "openai-compatible"
  base_url: "http://localhost:11434/v1"
  model_name: "qwen2.5-7b-4k:latest"
  temperature: 0.0
  max_completion_tokens: 256
schemas:
  TechKeywords: "schemas.tech_keywords:TechKeywords"
  RandomName: "schemas.tech_keywords:RandomName"
chain:
  - id: generate-random-name
    kind: structured
    output_schema: RandomName
    prompt_section: step:generate-random-name
  - id: tech-keyword-extraction
    kind: structured
    output_schema: TechKeywords
    prompt_section: step:tech-keyword-extraction

output:
  format: json
---

## step:generate-random-name

Generate a random first name.
Return only valid JSON with this exact shape:
{"name":"\<first\_name>"}
No markdown, no explanation, no extra keys.

## step:tech-keyword-extraction

You extract programming languages and technical skills from candidate writeups.
Return JSON with keys: `computer_languages` (array of strings), `databases` (array of strings), `other` (array of strings).
Use lowercase values and unique entries only.
