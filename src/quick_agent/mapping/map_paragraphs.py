"""Paragraph-first chunk mapper with sentence fallback."""

from __future__ import annotations

import re

from quick_agent.mapping.map_base import MapBase
from quick_agent.models.content_processing_spec import ChunkProcessingSpec


class MapParagraphs(MapBase):
    def build_chunks(self, text: str, map_config: ChunkProcessingSpec) -> list[str]:
        max_chunk_tokens = map_config.max_chunk_tokens
        paragraphs = self._split_paragraphs(text)
        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0
        index = 0
        while index < len(paragraphs):
            paragraph = paragraphs[index].strip()
            paragraph_tokens = self._encoding.encode(paragraph)
            paragraph_size = len(paragraph_tokens)
            if paragraph_size > max_chunk_tokens:
                if current_parts:
                    chunks.append("\n\n".join(current_parts).strip())
                    current_parts = []
                    current_tokens = 0
                chunks.extend(
                    self._split_paragraph_with_sentences(paragraph, max_chunk_tokens)
                )
                index += 1
                continue
            if current_tokens + paragraph_size <= max_chunk_tokens:
                current_parts.append(paragraph)
                current_tokens += paragraph_size
            else:
                if current_parts:
                    chunks.append("\n\n".join(current_parts).strip())
                current_parts = [paragraph]
                current_tokens = paragraph_size
            index += 1
        if current_parts:
            chunks.append("\n\n".join(current_parts).strip())
        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        entries = text.split("\n\n")
        paragraphs: list[str] = []
        index = 0
        while index < len(entries):
            entry = entries[index].strip()
            if entry:
                paragraphs.append(entry)
            index += 1
        if not paragraphs and text.strip():
            paragraphs.append(text.strip())
        return paragraphs

    def _split_paragraph_with_sentences(
        self, paragraph: str, max_chunk_tokens: int
    ) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        chunks: list[str] = []
        current = ""
        index = 0
        while index < len(sentences):
            sentence = sentences[index].strip()
            if not sentence:
                index += 1
                continue
            candidate = sentence if not current else f"{current} {sentence}"
            candidate_tokens = self._encoding.encode(candidate)
            if len(candidate_tokens) <= max_chunk_tokens:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                sentence_tokens = self._encoding.encode(sentence)
                if len(sentence_tokens) > max_chunk_tokens:
                    chunks.extend(
                        self._split_token_ids(sentence_tokens, max_chunk_tokens)
                    )
                    current = ""
                else:
                    current = sentence
            index += 1
        if current:
            chunks.append(current.strip())
        return chunks
