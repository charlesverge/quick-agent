# Chunking

This document defines chunk-processing examples for large-context input handling.

Input file used in all examples:
`examples/sample_25_50_25/input_6000_tokens.txt`

## `map_chunks`

Use fixed-size chunk splitting.

```yaml
content_processing:
  chunk_processing:
    mode: map_chunks
    provider: semchunks
    max_chunk_tokens: 1200
    overlap_percent: 10
```

Run:

```bash
quick-agent --agent large-context-map-chunks --input examples/sample_25_50_25/input_6000_tokens.txt
```

## `map_paragraphs`

Use paragraph-first chunk splitting with sentence fallback for oversized paragraphs.

```yaml
content_processing:
  chunk_processing:
    mode: map_paragraphs
    provider: semchunks
    max_chunk_tokens: 1200
    overlap_percent: 10
```

Run:

```bash
quick-agent --agent large-context-map-paragraphs --input examples/sample_25_50_25/input_6000_tokens.txt
```

## `semchunks` provider with explicit overlap tokens

Use provider-specific chunking with direct overlap token control.

```yaml
content_processing:
  chunk_processing:
    mode: map_chunks
    provider: semchunks
    max_chunk_tokens: 1200
    overlap_token: 120
```

Run:

```bash
quick-agent --agent large-context-map-semchunks --input examples/sample_25_50_25/input_6000_tokens.txt
```

## Output contract

Chunk map output remains:

```json
{"items": ["<raw chunk text>", "<raw chunk text>"]}
```
