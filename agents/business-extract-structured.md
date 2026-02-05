---
name: "Business Extract Structured"
description: "Extract company name, location, and summary into structured JSON."
model:
  provider: "openai-compatible"
  base_url: "http://localhost:11434/v1"
  api_key_env: "OPENAI_API_KEY"
  model_name: "llama3"
schemas:
  BusinessSummary: "quick_agent.schemas.outputs:BusinessSummary"
chain:
  - id: company_name
    kind: text
    prompt_section: step:company_name
  - id: location
    kind: text
    prompt_section: step:location
  - id: summary
    kind: structured
    prompt_section: step:summary
    output_schema: BusinessSummary
output:
  format: "json"
  file: "out/business_extract_structured.json"
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

Return a JSON object with:
- `company_name`
- `location`
- `summary` (one sentence)

Use `state.steps.company_name` and `state.steps.location` for the fields if available.
