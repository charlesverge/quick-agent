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
