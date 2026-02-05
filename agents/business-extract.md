---
name: "Business Extract"
description: "Extract company name, location, and a short summary from a business description."
model:
  provider: "openai-compatible"
  base_url: "http://localhost:11434/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "llama3"
chain:
  - id: company_name
    kind: text
    prompt_section: step:company_name
  - id: location
    kind: text
    prompt_section: step:location
  - id: summary
    kind: text
    prompt_section: step:summary
output:
  format: "markdown"
  file: "out/business_extract.md"
---

## Instructions

Extract structured details from the input description.

## step:company_name

Extract the company name from the input description.
Return only the company name.

## step:location

Extract the location from the input description.
If a city and region are present, include both.
Return only the location.

## step:summary

Write one sentence summarizing the business.
Use `state.steps.company_name` and `state.steps.location` if available.
