"""Rough token estimation helpers used by the context policy."""

from __future__ import annotations

import json
from typing import Any


CHARS_PER_ESTIMATED_TOKEN = 4


def estimate_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except TypeError:
            text = str(value)
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_ESTIMATED_TOKEN - 1) // CHARS_PER_ESTIMATED_TOKEN)


def estimate_record_tokens(record: dict) -> int:
    cached = record.get("estimated_tokens")
    if isinstance(cached, int) and cached >= 0:
        return cached

    role = record.get("role")
    record_type = record.get("type")

    if role in {"user", "assistant"}:
        return estimate_tokens(record.get("content", ""))

    if record_type == "function_call":
        return estimate_tokens(record.get("name", "")) + estimate_tokens(record.get("arguments", ""))

    if record_type == "function_call_output":
        return estimate_tokens(record.get("output", ""))

    if record_type == "context_summary":
        return estimate_tokens(record.get("summary", {}))

    return estimate_tokens(record)


def estimate_records_tokens(records: list[dict]) -> int:
    return sum(estimate_record_tokens(record) for record in records)


__all__ = ["CHARS_PER_ESTIMATED_TOKEN", "estimate_record_tokens", "estimate_records_tokens", "estimate_tokens"]
