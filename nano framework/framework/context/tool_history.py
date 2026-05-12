"""Tool-history compaction for Responses function-call records."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from ..range_preview import build_range_preview
from .context_policy import ContextPolicy, DEFAULT_CONTEXT_POLICY
from .token_budget import estimate_tokens


TIME_BASED_MC_CLEARED_MESSAGE = "历史工具结果过长，以下仅保留范围式预览。"
HISTORY_ARG_PREVIEW_MARKER = "[history_tool_args_preview]"

_KEEP_ARGUMENT_KEYS = {
    "path",
    "paths",
    "file_path",
    "query",
    "kind",
    "target",
    "log_path",
}
_RANGE_PREVIEW_KEYS = {"content", "search", "replace", "patch", "payload_json"}


@dataclass(frozen=True)
class ToolHistoryDiagnostics:
    compactable_results: int = 0
    kept_results: int = 0
    cleared_results: int = 0
    tokens_saved: int = 0


def clear_old_tool_results(
    records: list[dict],
    policy: ContextPolicy = DEFAULT_CONTEXT_POLICY,
) -> tuple[list[dict], ToolHistoryDiagnostics]:
    """Keep recent tool turns verbatim and summarize only older compactable history."""
    compactable_call_ids = _collect_compactable_call_ids(records, policy)
    compactable_function_call_ids = _collect_compactable_function_call_ids(records, policy)
    recent_turn_call_ids = _collect_recent_tool_turn_call_ids(records, policy)

    keep_recent_results = max(0, policy.keep_recent_tool_results)
    keep_recent_argument_calls = max(0, policy.keep_recent_tool_argument_calls)

    older_output_call_ids = [call_id for call_id in compactable_call_ids if call_id not in recent_turn_call_ids]
    older_argument_call_ids = [
        call_id for call_id in compactable_function_call_ids if call_id not in recent_turn_call_ids
    ]

    keep_output_ids = recent_turn_call_ids | (
        set(older_output_call_ids[-keep_recent_results:]) if keep_recent_results else set()
    )
    keep_argument_ids = recent_turn_call_ids | (
        set(older_argument_call_ids[-keep_recent_argument_calls:]) if keep_recent_argument_calls else set()
    )

    call_id_to_tool: dict[str, str] = {}
    projected: list[dict] = []
    cleared_results = 0
    tokens_saved = 0

    for record in records:
        record_type = record.get("type")

        if record_type == "function_call":
            tool_name = str(record.get("name") or "")
            call_id = record.get("call_id")
            if isinstance(call_id, str) and tool_name:
                call_id_to_tool[call_id] = tool_name

            if _is_compactable_tool(tool_name, policy) and call_id not in keep_argument_ids:
                projected.append(_snip_function_call(record, tool_name, policy))
            else:
                projected.append(record)
            continue

        if record_type == "function_call_output":
            call_id = record.get("call_id")
            tool_name = call_id_to_tool.get(call_id) if isinstance(call_id, str) else None
            if tool_name and _is_compactable_tool(tool_name, policy) and call_id not in keep_output_ids:
                original_tokens = estimate_tokens(record.get("output", ""))
                summarized = _summarize_function_output(record, tool_name, policy)
                cleared_results += 1
                tokens_saved += max(0, original_tokens - estimate_tokens(summarized.get("output", "")))
                projected.append(summarized)
            else:
                projected.append(record)
            continue

        projected.append(record)

    return projected, ToolHistoryDiagnostics(
        compactable_results=len(compactable_call_ids),
        kept_results=max(0, len(compactable_call_ids) - cleared_results),
        cleared_results=cleared_results,
        tokens_saved=tokens_saved,
    )


def sanitize_records_for_context(records: list[dict]) -> list[dict]:
    """Drop malformed records that should not be sent back to the Responses API."""
    output_call_ids = {
        record.get("call_id")
        for record in records
        if record.get("type") == "function_call_output" and isinstance(record.get("call_id"), str)
    }
    kept: list[dict] = []
    kept_call_ids: set[str] = set()
    completed_call_ids: set[str] = set()

    for record in records:
        role = record.get("role")
        record_type = record.get("type")

        if role in {"user", "assistant"}:
            if record.get("content"):
                kept.append(record)
            continue

        if record_type == "function_call":
            call_id = record.get("call_id")
            name = record.get("name")
            if not (isinstance(call_id, str) and call_id and isinstance(name, str) and name):
                continue
            if call_id in kept_call_ids or call_id not in output_call_ids:
                continue
            kept_call_ids.add(call_id)
            kept.append(record)
            continue

        if record_type == "function_call_output":
            call_id = record.get("call_id")
            if not isinstance(call_id, str) or call_id not in kept_call_ids:
                continue
            if call_id in completed_call_ids:
                continue
            completed_call_ids.add(call_id)
            kept.append(record)
            continue

        kept.append(record)

    return kept


def _collect_compactable_call_ids(records: list[dict], policy: ContextPolicy) -> list[str]:
    call_id_to_tool: dict[str, str] = {}
    compactable_call_ids: list[str] = []

    for record in records:
        record_type = record.get("type")
        if record_type == "function_call":
            tool_name = str(record.get("name") or "")
            call_id = record.get("call_id")
            if isinstance(call_id, str) and tool_name:
                call_id_to_tool[call_id] = tool_name
            continue

        if record_type == "function_call_output":
            call_id = record.get("call_id")
            tool_name = call_id_to_tool.get(call_id) if isinstance(call_id, str) else None
            if tool_name and _is_compactable_tool(tool_name, policy):
                compactable_call_ids.append(call_id)

    return compactable_call_ids


def _collect_compactable_function_call_ids(records: list[dict], policy: ContextPolicy) -> list[str]:
    call_ids: list[str] = []
    for record in records:
        if record.get("type") != "function_call":
            continue
        tool_name = str(record.get("name") or "")
        call_id = record.get("call_id")
        if isinstance(call_id, str) and tool_name and _is_compactable_tool(tool_name, policy):
            call_ids.append(call_id)
    return call_ids


def _collect_recent_tool_turn_call_ids(records: list[dict], policy: ContextPolicy) -> set[str]:
    turn_limit = max(0, policy.keep_recent_tool_turns)
    if turn_limit == 0:
        return set()

    tool_turns = _collect_tool_turn_call_ids(records)
    kept: set[str] = set()
    for call_ids in tool_turns[-turn_limit:]:
        kept.update(call_ids)
    return kept


def _collect_tool_turn_call_ids(records: list[dict]) -> list[set[str]]:
    turns: list[set[str]] = []
    current_call_ids: set[str] = set()
    open_tool_call_ids: set[str] = set()

    def flush_tool_turn() -> None:
        nonlocal current_call_ids, open_tool_call_ids
        if current_call_ids:
            turns.append(set(current_call_ids))
            current_call_ids = set()
            open_tool_call_ids = set()

    for record in records:
        record_type = record.get("type")
        if record_type in {"context_compact_boundary", "context_summary", "context_compact_failure", "token_usage"}:
            continue

        if record_type == "function_call":
            if current_call_ids and not open_tool_call_ids:
                flush_tool_turn()
            call_id = record.get("call_id")
            if isinstance(call_id, str) and call_id:
                current_call_ids.add(call_id)
                open_tool_call_ids.add(call_id)
            continue

        if record_type == "function_call_output":
            call_id = record.get("call_id")
            if isinstance(call_id, str) and call_id:
                current_call_ids.add(call_id)
                open_tool_call_ids.discard(call_id)
            continue

        flush_tool_turn()

    flush_tool_turn()
    return turns


def _is_compactable_tool(tool_name: str, policy: ContextPolicy) -> bool:
    return tool_name in policy.compactable_tools and tool_name not in policy.control_tools


def _snip_function_call(record: dict, tool_name: str, policy: ContextPolicy) -> dict:
    arguments = record.get("arguments", "")
    if not isinstance(arguments, str) or not arguments:
        return record

    payload = _try_json_loads(arguments)
    if isinstance(payload, dict):
        compact_payload = (
            _compact_submit_structured_result_args(payload, policy)
            if tool_name == "submit_structured_result"
            else _compact_generic_args(payload, policy)
        )
        compact_text = json.dumps(compact_payload, ensure_ascii=False, indent=2)
    else:
        compact_text = arguments

    summary = "\n".join(
        [
            HISTORY_ARG_PREVIEW_MARKER,
            f"tool: {tool_name}",
            build_range_preview(compact_text, max(1, policy.argument_preview_chars)),
        ]
    )
    return _replace_arguments(record, summary)


def _summarize_function_output(record: dict, tool_name: str, policy: ContextPolicy) -> dict:
    output = record.get("output", "")
    text = output if isinstance(output, str) else str(output)
    limit = max(0, policy.old_tool_output_preview_chars)
    if len(text) <= limit:
        summary = text
    else:
        preview = build_range_preview(text, limit)
        summary = f"{TIME_BASED_MC_CLEARED_MESSAGE}\ntool: {tool_name}\n{preview}"
    return {
        **record,
        "output": summary,
        "estimated_tokens": estimate_tokens(summary),
    }


def _compact_submit_structured_result_args(payload: dict[str, Any], policy: ContextPolicy) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    if "kind" in payload:
        compact["kind"] = payload.get("kind")

    payload_json = payload.get("payload_json")
    if isinstance(payload_json, str):
        compact["payload_json"] = _compact_value(payload_json, policy)
    elif payload_json is not None:
        compact["payload_json"] = build_range_preview(str(payload_json), max(1, policy.argument_preview_chars))

    for key, value in payload.items():
        if key not in compact and key in _KEEP_ARGUMENT_KEYS:
            compact[key] = _compact_named_value(key, value, policy)
    return compact


def _compact_generic_args(payload: dict[str, Any], policy: ContextPolicy) -> dict[str, Any]:
    return {key: _compact_named_value(key, value, policy) for key, value in payload.items()}


def _compact_value(value: Any, policy: ContextPolicy) -> Any:
    if isinstance(value, str):
        if len(value) > policy.argument_preview_chars:
            return build_range_preview(value, max(1, policy.argument_preview_chars))
        return value
    if isinstance(value, list):
        return [_compact_value(item, policy) for item in value]
    if isinstance(value, dict):
        return {key: _compact_value(item, policy) for key, item in value.items()}
    return value


def _compact_named_value(key: str, value: Any, policy: ContextPolicy) -> Any:
    if isinstance(value, str) and key in _RANGE_PREVIEW_KEYS:
        return build_range_preview(value, max(1, policy.argument_preview_chars))
    return _compact_value(value, policy)


def _replace_arguments(record: dict, arguments: str) -> dict:
    return {
        **record,
        "arguments": arguments,
        "estimated_tokens": estimate_tokens(record.get("name", "")) + estimate_tokens(arguments),
    }


def _snip_text(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else str(value)
    if len(text) <= limit:
        return text
    return build_range_preview(text, limit)


def _try_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


__all__ = [
    "HISTORY_ARG_PREVIEW_MARKER",
    "TIME_BASED_MC_CLEARED_MESSAGE",
    "ToolHistoryDiagnostics",
    "clear_old_tool_results",
    "sanitize_records_for_context",
]
