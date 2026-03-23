import pytest

from quick_agent.models.content_processing_spec import SampleSpec
from quick_agent.samplers.simple_ratios import SampleRatios


def test_sample_validation_allows_zero_ratios() -> None:
    sampler = SampleRatios()
    config = SampleSpec(ratios=(0, 100, 0), max_chunk_tokens=10)

    result = sampler.run(
        "one two three four five six seven eight nine ten eleven", config
    )

    assert result


def test_sample_validation_rejects_zero_ratio_sum() -> None:
    sampler = SampleRatios()
    config = SampleSpec(ratios=(0, 0, 0), max_chunk_tokens=10)

    with pytest.raises(ValueError, match="sample.ratios sum must be greater than 0."):
        sampler.run("one two three four", config)


def test_sample_ratio_center_only() -> None:
    sampler = SampleRatios()
    config = SampleSpec(ratios=(0, 100, 0), max_chunk_tokens=4)
    text = "one two three four five six seven eight nine ten"

    result = sampler.run(text, config)

    assert result == "four five six seven"


def test_sample_ratio_head_footer_only() -> None:
    sampler = SampleRatios()
    config = SampleSpec(ratios=(50, 0, 50), max_chunk_tokens=4)
    text = "one two three four five six seven eight nine ten"

    result = sampler.run(text, config)

    assert result == "one two nine ten"


def test_sample_ratio_header_only() -> None:
    sampler = SampleRatios()
    config = SampleSpec(ratios=(100, 0, 0), max_chunk_tokens=4)
    text = "one two three four five six seven eight nine ten"

    result = sampler.run(text, config)

    assert result == "one two three four"


def test_sample_passthrough_when_text_fits_budget() -> None:
    sampler = SampleRatios()
    config = SampleSpec(ratios=(25, 50, 25), max_chunk_tokens=10)
    text = "one two three four five six"

    result = sampler.run(text, config)

    assert result == text


def test_sample_enforce_limit_trims_to_budget() -> None:
    sampler = SampleRatios()

    result = sampler.enforce_limit("one two three four five six seven eight", 5)

    assert len(sampler._encoding.encode(result)) == 5


def test_sample_three_x_input_center_only() -> None:
    sampler = SampleRatios()
    config = SampleSpec(ratios=(0, 100, 0), max_chunk_tokens=4)
    text = "one two three four five six seven eight nine ten eleven twelve"

    result = sampler.run(text, config)

    assert len(sampler._encoding.encode(text)) == 12
    assert len(sampler._encoding.encode(result)) == 4


def test_sample_three_x_input_head_footer_only() -> None:
    sampler = SampleRatios()
    config = SampleSpec(ratios=(50, 0, 50), max_chunk_tokens=4)
    text = "one two three four five six seven eight nine ten eleven twelve"

    result = sampler.run(text, config)

    assert len(sampler._encoding.encode(text)) == 12
    assert len(sampler._encoding.encode(result)) == 4
