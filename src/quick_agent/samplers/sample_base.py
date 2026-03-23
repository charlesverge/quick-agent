"""Sample processor base contract."""

from __future__ import annotations

import tiktoken

from quick_agent.models.content_processing_spec import SampleSpec


class SampleBase:
    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")

    def run(self, text: str, sample_config: SampleSpec) -> str:
        raise NotImplementedError

    def validate(self, sample_config: SampleSpec) -> None:
        if sample_config.max_chunk_tokens <= 0:
            raise ValueError("sample.max_chunk_tokens must be greater than 0.")
        if len(sample_config.ratios) != 3:
            raise ValueError("sample.ratios must contain exactly 3 values.")
        if (
            sample_config.ratios[0] < 0
            or sample_config.ratios[1] < 0
            or sample_config.ratios[2] < 0
        ):
            raise ValueError("sample.ratios values must be non-negative.")
        if sum(sample_config.ratios) <= 0:
            raise ValueError("sample.ratios sum must be greater than 0.")

    def sample_text(self, text: str, sample_config: SampleSpec) -> str:
        token_budget = sample_config.max_chunk_tokens
        source_tokens = self._encoding.encode(text)
        source_count = len(source_tokens)
        if source_count <= token_budget:
            return text

        ratios = sample_config.ratios
        ratio_sum = sum(ratios)
        head_count = (token_budget * ratios[0]) // ratio_sum
        middle_count = (token_budget * ratios[1]) // ratio_sum
        footer_count = (token_budget * ratios[2]) // ratio_sum

        assigned = head_count + middle_count + footer_count
        remainder = token_budget - assigned
        if remainder > 0:
            if ratios[0] > 0:
                head_count += 1
                remainder -= 1
        if remainder > 0:
            if ratios[1] > 0:
                middle_count += 1
                remainder -= 1
        if remainder > 0:
            if ratios[2] > 0:
                footer_count += remainder

        head_tokens = source_tokens[:head_count] if head_count > 0 else []

        middle_tokens: list[int] = []
        if middle_count > 0:
            middle_start = (source_count - middle_count) // 2
            middle_end = middle_start + middle_count
            middle_tokens = source_tokens[middle_start:middle_end]

        footer_tokens = source_tokens[-footer_count:] if footer_count > 0 else []
        sampled_tokens = head_tokens + middle_tokens + footer_tokens
        return self._encoding.decode(sampled_tokens).strip()

    def enforce_limit(self, text: str, max_chunk_tokens: int) -> str:
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_chunk_tokens:
            return text
        return self._encoding.decode(tokens[:max_chunk_tokens]).strip()
