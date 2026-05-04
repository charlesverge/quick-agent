"""QuickAgent-specific exceptions."""

from __future__ import annotations


class QuickAgentException(Exception):
    """Base exception for QuickAgent runtime failures."""


class QuickAgentToolsNotSupportedException(QuickAgentException):
    """Raised when a model rejects tool usage."""

    def __init__(self, *, model_name: str, message: str) -> None:
        super().__init__(message)
        self.model_name = model_name
        self.message = message


class QuickAgentChatNotSupportedException(QuickAgentException):
    """Raised when a model does not support chat completions."""

    def __init__(self, *, model_name: str, message: str) -> None:
        super().__init__(message)
        self.model_name = model_name
        self.message = message

class QuickAgentLLMTemporaryException(QuickAgentException):
    """Raised when a model returns a temporary error, such as a rate limit error, invalid json output.
    This indicates that the request may succeed if retried after a delay.
    """

    def __init__(self, *, message: str, output: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.output = output