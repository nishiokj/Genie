from __future__ import annotations

from typing import Any


def _nonempty_string(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _positive_number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    return None
