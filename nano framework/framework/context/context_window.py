"""Recent-window selection with Responses API invariants."""

from __future__ import annotations

from .context_policy import ContextPolicy, DEFAULT_CONTEXT_POLICY
from .token_budget import estimate_record_tokens


def split_for_compaction(
    records: list[dict],
    policy: ContextPolicy = DEFAULT_CONTEXT_POLICY,
) -> tuple[list[dict], list[dict]]:
    """Split records into summary prefix and preserved recent tail."""
    if not records:
        return [], []

    start_index = _calculate_tail_start_index(records, policy)
    prefix = records[:start_index]
    tail = records[start_index:]
    return _preserve_tool_pairs(prefix, tail)


def _calculate_tail_start_index(records: list[dict], policy: ContextPolicy) -> int:
    start_index = len(records)
    total_tokens = 0
    text_messages = 0

    for index in range(len(records) - 1, -1, -1):
        record = records[index]
        record_tokens = estimate_record_tokens(record)
        if total_tokens and total_tokens + record_tokens > policy.max_preserved_tokens:
            break

        total_tokens += record_tokens
        if _has_text_content(record):
            text_messages += 1
        start_index = index

        if total_tokens >= policy.min_preserved_tokens and text_messages >= policy.min_preserved_text_messages:
            break

    return start_index


def _has_text_content(record: dict) -> bool:
    role = record.get("role")
    content = record.get("content")
    return role in {"user", "assistant"} and isinstance(content, str) and bool(content.strip())


def _preserve_tool_pairs(prefix: list[dict], tail: list[dict]) -> tuple[list[dict], list[dict]]:
    """Move needed function_call records into the tail when the split cuts a pair."""
    if not prefix or not tail:
        return prefix, tail

    tail_output_ids = {
        record.get("call_id")
        for record in tail
        if record.get("type") == "function_call_output" and isinstance(record.get("call_id"), str)
    }
    tail_call_ids = {
        record.get("call_id")
        for record in tail
        if record.get("type") == "function_call" and isinstance(record.get("call_id"), str)
    }
    needed_call_ids = tail_output_ids - tail_call_ids
    if not needed_call_ids:
        return prefix, tail

    moved: list[dict] = []
    kept_prefix: list[dict] = []
    for record in prefix:
        if record.get("type") == "function_call" and record.get("call_id") in needed_call_ids:
            moved.append(record)
        else:
            kept_prefix.append(record)

    return kept_prefix, moved + tail


__all__ = ["split_for_compaction"]

