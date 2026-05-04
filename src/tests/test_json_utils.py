import pytest

from quick_agent.exceptions import QuickAgentLLMTemporaryException
from quick_agent.json_utils import extract_extract_brackets


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{{"foo": "bar"}', '{"foo": "bar"}'),
        ('{"{"foo": "bar"}', '{"foo": "bar"}'),
        (r'{\"{"foo": "bar"}', '{"foo": "bar"}'),
        ('{{a": 1}', '{"a": 1}'),
        ('{"{a": 1}', '{"a": 1}'),
        ('{{a": 1}', '{"a": 1}'),
        ('{\n{a": 1}', '{"a": 1}'),
        ('{\n"{a": 1}', '{"a": 1}'),
        ('{\n{"foo": "bar"}', '{"foo": "bar"}'),
        ('{\n"{"foo": "bar"}', '{"foo": "bar"}'),
        (r'{\n\"{"foo": "bar"}', '{"foo": "bar"}'),
        (r'{"{"selected_source_urls":["http"]}', '{"selected_source_urls":["http"]}'),
        (r'[]{"locations":[]}', '{"locations":[]}'),
    ],
)
def test_extract_extract_brackets_handles_extra_bracket_variants(
    text: str, expected: str
) -> None:
    result = extract_extract_brackets(text)

    assert result == expected


def test_extract_extract_brackets_raises_for_non_extra_bracket_text() -> None:
    text = '{"foo": "bar"}'

    with pytest.raises(QuickAgentLLMTemporaryException) as exc_info:
        extract_extract_brackets(text)

    assert exc_info.value.message == 'extract_extract_brackets failed.'
