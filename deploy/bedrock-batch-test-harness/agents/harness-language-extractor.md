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
output:
  format: json
  output_schema: TechKeywords
---

# System Prompt

You extract programming languages and technical skills from candidate writeups.
Return JSON with keys: computer_languages (array of strings), databases (array of strings), other (array of strings).
Use lowercase values and unique entries only.
