"""Base class for map chunk-processing."""

from __future__ import annotations

import tiktoken

from quick_agent.models.content_processing_spec import ChunkProcessingSpec


class MapBase:
    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def run(self, text: str, map_config: ChunkProcessingSpec) -> list[str]:
        self.validate(map_config)
        chunks = self.build_chunks(text, map_config)
        chunks = self.apply_overlap(chunks, map_config)
        return self.normalize(chunks, map_config)

    def validate(self, map_config: ChunkProcessingSpec) -> None:
        if map_config.max_chunk_tokens <= 0:
            raise ValueError(
                "chunk_processing.max_chunk_tokens must be greater than 0."
            )
        if map_config.overlap_percent < 0 or map_config.overlap_percent > 20:
            raise ValueError("chunk_processing.overlap_percent must be in 0..20.")
        if map_config.overlap_token is not None and map_config.overlap_token < 0:
            raise ValueError("chunk_processing.overlap_token must be non-negative.")

    def build_chunks(self, text: str, map_config: ChunkProcessingSpec) -> list[str]:
        raise NotImplementedError

    def apply_overlap(
        self, chunks: list[str], map_config: ChunkProcessingSpec
    ) -> list[str]:
        if not chunks:
            return []
        overlap_count = self._resolve_overlap(map_config)
        if overlap_count <= 0:
            return chunks
        overlapped: list[str] = [chunks[0]]
        max_chunk_tokens = map_config.max_chunk_tokens
        index = 1
        while index < len(chunks):
            previous_tokens = self._encoding.encode(chunks[index - 1])
            current_tokens = self._encoding.encode(chunks[index])
            prefix_tokens = previous_tokens[-overlap_count:]
            merged_tokens = prefix_tokens + current_tokens
            if len(merged_tokens) > max_chunk_tokens:
                merged_tokens = merged_tokens[:max_chunk_tokens]
            overlapped.append(self._encoding.decode(merged_tokens).strip())
            index += 1
        return overlapped

    def normalize(
        self, chunks: list[str], map_config: ChunkProcessingSpec
    ) -> list[str]:
        normalized: list[str] = []
        max_chunk_tokens = map_config.max_chunk_tokens
        max_output_items = map_config.max_output_items
        max_output_tokens = map_config.max_output_tokens
        used_output_tokens = 0
        index = 0
        while index < len(chunks):
            if max_output_items is not None and len(normalized) >= max_output_items:
                break
            chunk = chunks[index]
            chunk_tokens = self._encoding.encode(chunk)
            if len(chunk_tokens) > max_chunk_tokens:
                chunk_tokens = chunk_tokens[:max_chunk_tokens]
            if max_output_tokens is not None:
                remaining = max_output_tokens - used_output_tokens
                if remaining <= 0:
                    break
                if len(chunk_tokens) > remaining:
                    chunk_tokens = chunk_tokens[:remaining]
                used_output_tokens += len(chunk_tokens)
            trimmed = self._encoding.decode(chunk_tokens).strip()
            if trimmed:
                normalized.append(trimmed)
            if (
                max_output_tokens is not None
                and used_output_tokens >= max_output_tokens
            ):
                break
            index += 1
        return normalized

    def _resolve_overlap(self, map_config: ChunkProcessingSpec) -> int:
        if map_config.overlap_token is not None:
            return map_config.overlap_token
        return (map_config.max_chunk_tokens * map_config.overlap_percent) // 100

    def _split_token_ids(self, token_ids: list[int], chunk_size: int) -> list[str]:
        if not token_ids:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(token_ids):
            end = start + chunk_size
            chunks.append(self._encoding.decode(token_ids[start:end]).strip())
            start = end
        return chunks
