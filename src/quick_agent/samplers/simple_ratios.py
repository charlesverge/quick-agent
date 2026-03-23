"""Ratio-based sample processor."""

from __future__ import annotations

from quick_agent.models.content_processing_spec import SampleSpec
from quick_agent.samplers.sample_base import SampleBase


class SampleRatios(SampleBase):
    def run(self, text: str, sample_config: SampleSpec) -> str:
        self.validate(sample_config)
        sampled_text = self.sample_text(text, sample_config)
        return self.enforce_limit(sampled_text, sample_config.max_chunk_tokens)
