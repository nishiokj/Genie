from __future__ import annotations

from typing import Any


class ProviderError(RuntimeError):
    pass


class ProviderStructuredOutputError(ProviderError):
    def __init__(self, message: str, *, raw_content: str = "", parsing_error: Any = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content
        self.parsing_error = parsing_error
