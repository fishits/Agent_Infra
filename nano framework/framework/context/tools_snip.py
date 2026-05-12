"""Compatibility entrypoints for context tool snipping."""

from __future__ import annotations

from .tool_history import clear_old_tool_results, sanitize_records_for_context


def snip_tool_records_for_context(records: list[dict]) -> list[dict]:
    """Return a context-facing view with old compactable tool results cleared."""
    projected, _diagnostics = clear_old_tool_results(records)
    return projected


__all__ = ["sanitize_records_for_context", "snip_tool_records_for_context"]
