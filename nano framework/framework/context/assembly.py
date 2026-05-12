"""Context assembly for Responses API input items.

This module follows a clear context pipeline:
build -> trim tools -> maybe compact via one summary invoke -> project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
from typing import Any

from openai import OpenAI

from .. import llm as _llm_env_loader  # noqa: F401 - import for .env side-effect
from ..todo_state import render_todo_state_for_context
from .compact_prompt import COMPACT_PROMPT
from .context_policy import DEFAULT_CONTEXT_POLICY
from .context_window import split_for_compaction
from ..range_preview import build_range_preview
from .store import append_items, read_latest_compaction, read_latest_token_usage, read_records
from .tool_history import ToolHistoryDiagnostics, clear_old_tool_results, sanitize_records_for_context


_CHARS_PER_ESTIMATED_TOKEN = 4
_CONTEXT_BUDGET_TOKENS = DEFAULT_CONTEXT_POLICY.context_window_tokens
_COMPACT_TRIGGER_TOKENS = DEFAULT_CONTEXT_POLICY.autocompact_threshold_tokens
_COMPACT_MODEL_CANDIDATES = [
    model_name
    for model_name in (
        os.getenv("OPENAI_COMPACT_MODEL"),
        os.getenv("OPENAI_MODEL"),
        "gpt-5.4-mini",
        "gpt-5.2",
        "gpt-5.4",
        "gpt-5.5",
    )
    if model_name
]

_COMPACT_PROMPT = COMPACT_PROMPT
_MAX_RESTORED_CONTEXT_FILES = 10
_INTERNAL_CONTEXT_MARKERS = ("_snipped_for_context", "_omitted", "[history_tool_args_preview]")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_ONLYTEAM_CONSTITUTION_PATH = _REPO_ROOT / "OnlyTeam" / "constitution.md"


@dataclass
class ContextState:
    run_id: str
    node_name: str
    records: list[dict]
    fixed_context: list[dict] = field(default_factory=list)
    todo_context: dict | None = None
    compressed_context: dict | None = None
    working_context: list[dict] = field(default_factory=list)
    compact_prefix: list[dict] = field(default_factory=list)
    compact_tail: list[dict] = field(default_factory=list)
    restored_context: list[dict] = field(default_factory=list)
    compacted_count: int = 0
    fixed_tokens: int = 0
    compressed_tokens: int = 0
    working_tokens: int = 0
    tool_tokens: int = 0
    total_tokens: int = 0
    api_input_tokens: int | None = None
    api_output_tokens: int | None = None
    api_total_tokens: int | None = None
    needs_compact: bool = False
    estimated_tokens: int = 0
    compact_failure_count: int = 0
    tool_history_diagnostics: ToolHistoryDiagnostics = field(default_factory=ToolHistoryDiagnostics)


def build_context(run_id: str, node_name: str) -> ContextState:
    """输入: `run_id`/`node_name`, 用来拿到这个 node 的全部本地历史 records。
    处理: 读取历史, 识别最近一次压缩点, 然后把上下文拆成固定区、压缩区、工作区这几个状态字段。
    输出: 一个 `ContextState`, 里面已经装好了 `compressed_context`、`working_context` 和预算基础数据。"""
    records = read_records(run_id, node_name)
    records = [_with_estimated_tokens(record) for record in records]
    latest_compaction = read_latest_compaction(run_id, node_name)
    latest_usage = read_latest_token_usage(run_id, node_name)
    compressed_context = latest_compaction.get("summary")
    restored_context = latest_compaction.get("restored_context") or []
    compacted_count = int(latest_compaction.get("compacted_count", 0) or 0)
    source_records = [
        record
        for record in records
        if record.get("type") not in {"context_compact_boundary", "context_summary", "context_compact_failure", "token_usage"}
    ]
    working_records, tool_history_diagnostics = clear_old_tool_results(list(source_records[compacted_count:]))
    working_records = filter_records_for_model_context(working_records)
    todo_context = render_todo_state_for_context(run_id, node_name)

    return ContextState(
        run_id=run_id,
        node_name=node_name,
        records=records,
        fixed_context=_load_fixed_context(node_name),
        todo_context=todo_context,
        compressed_context=compressed_context,
        restored_context=restored_context if isinstance(restored_context, list) else [],
        working_context=working_records,
        compacted_count=compacted_count,
        api_input_tokens=latest_usage.get("input_tokens"),
        api_output_tokens=latest_usage.get("output_tokens"),
        api_total_tokens=latest_usage.get("total_tokens"),
        compact_failure_count=_count_recent_compact_failures(records),
        tool_history_diagnostics=tool_history_diagnostics,
    )


def _load_fixed_context(node_name: str) -> list[dict]:
    if not str(node_name or "").lower().startswith("onlyteam_"):
        return []

    if not _ONLYTEAM_CONSTITUTION_PATH.exists():
        return []

    content = _ONLYTEAM_CONSTITUTION_PATH.read_text(encoding="utf-8").strip()
    if not content:
        return []

    return [{"role": "user", "content": f"OnlyTeam 宪法\n\n{content}"}]


def trim_tool_results(state: ContextState) -> ContextState:
    """输入: 一个 `ContextState`, 里面已经有当前候选工作区 records。
    处理: 只对工作区里的超长 `function_call_output` 做轻量裁剪, 防止工具输出先把预算顶爆。
    输出: 还是同一个 `ContextState`, 但 `working_context` 会更轻, 预算字段也会顺手刷新。"""
    measure_context_budget(state)
    return state


def measure_context_budget(state: ContextState) -> ContextState:
    """输入: 一个已经分好区的 `ContextState`。
    处理: 分别统计固定区、压缩区、工作区、工具相关内容的大致 token 预算。
    输出: 还是这个 `ContextState`, 但会补齐 `fixed_tokens`、`working_tokens`、`tool_tokens`、`total_tokens` 等字段。"""
    state.fixed_tokens = _estimate_records_tokens(state.fixed_context)
    state.compressed_tokens = (
        _estimate_tokens(_render_summary_for_context(state.node_name, state.compressed_context))
        if state.compressed_context is not None
        else 0
    )
    state.working_tokens = _estimate_records_tokens(state.working_context)
    state.tool_tokens = sum(
        _estimate_record_tokens(record)
        for record in state.working_context
        if record.get("type") in {"function_call", "function_call_output"}
    )
    todo_tokens = _estimate_record_tokens(state.todo_context) if state.todo_context is not None else 0
    estimated_total = state.fixed_tokens + state.compressed_tokens + state.working_tokens + todo_tokens
    state.total_tokens = estimated_total
    state.estimated_tokens = estimated_total
    return state


def should_compact(state: ContextState) -> bool:
    """输入: 一个 `ContextState`, 重点关注里面刚统计出来的预算字段。
    处理: 按总预算和分区预算判断这轮是否应该进入 compact 流程, 这里只做判断不执行压缩。
    输出: 一个 `bool`; 同时会把 `state.needs_compact` 这个标记写回状态里。"""
    measure_context_budget(state)
    state.needs_compact = (
        state.total_tokens > _COMPACT_TRIGGER_TOKENS
        or state.working_tokens > _CONTEXT_BUDGET_TOKENS
    )
    return state.needs_compact


def run_compaction(state: ContextState) -> ContextState:
    """输入: 一个已经算好预算的 `ContextState`, 重点看 `needs_compact` 和当前工作区。
    处理: 如果不需要压缩就直接返回; 如果需要, 就执行 split -> summary invoke -> compact apply 这条完整链。
    输出: 一个新的 `ContextState`; 需要压缩时会把旧前缀替换成压缩区, 不需要时保持原样。"""
    if not state.needs_compact:
        return state
    if state.compact_failure_count >= DEFAULT_CONTEXT_POLICY.max_consecutive_compact_failures:
        state.needs_compact = False
        return state

    try:
        summary_text = invoke_compact_summary(state)
    except Exception as exc:
        _record_compact_failure(state, "exception", str(exc))
        state.compact_failure_count += 1
        state.needs_compact = False
        return state
    if not summary_text.strip():
        _record_compact_failure(state, "empty_summary", "")
        state.compact_failure_count += 1
        state.needs_compact = False
        return state

    state = compact_context(state, summary_text)
    measure_context_budget(state)
    should_compact(state)
    return state


def _record_compact_failure(state: ContextState, reason: str, detail: str) -> None:
    append_items(
        state.run_id,
        state.node_name,
        [
            {
                "type": "context_compact_failure",
                "created_at": _utc_now(),
                "reason": reason,
                "detail": detail[:1_000],
                "total_tokens": state.total_tokens,
                "working_tokens": state.working_tokens,
                "tool_tokens": state.tool_tokens,
            }
        ],
    )


def invoke_compact_summary(state: ContextState) -> str:
    """输入: 一个 `ContextState`, 默认从里面取出“要被压缩的前半段历史”。
    处理: 按 Claude 风格再发起一次无工具 summary invoke, 让模型把旧上下文写成结构化总结。
    输出: 一个 `str` 原始 summary 文本, 通常带 `<analysis>` 和 `<summary>` 包裹。"""
    prefix, tail = split_records_for_compaction(state.working_context)
    if not prefix:
        return ""

    client = OpenAI()
    compact_items = build_compaction_input_items(state, prefix)
    last_error: Exception | None = None
    response = None
    for model_name in _COMPACT_MODEL_CANDIDATES:
        try:
            response = client.responses.create(
                model=model_name,
                input=compact_items + [{"role": "user", "content": _COMPACT_PROMPT}],
                tools=[],
                stream=False,
                store=True,
            )
            break
        except Exception as exc:  # pragma: no cover - network/provider fallback
            last_error = exc

    if response is None:
        if last_error is not None:
            raise last_error
        return ""

    state.compact_prefix = prefix
    state.compact_tail = tail
    return _extract_response_text(response)


def compact_context(state: ContextState, summary_text: str) -> ContextState:
    """输入: 一个 `ContextState` 和一段已经生成好的 `summary_text`。
    处理: 解析总结, 落盘压缩边界与 summary 记录, 然后把旧前缀替换成压缩区, 保留最近工作区。
    输出: 更新后的 `ContextState`, 其中 `compressed_context` 和 `working_context` 都会变成压缩后的新状态。"""
    if not summary_text or not state.compact_prefix:
        return state

    summary = _parse_structured_summary(summary_text)
    restored_context = _build_restored_context_notes(state.compact_prefix, state.compact_tail)
    post_compact_tokens = _estimate_post_compact_tokens(state.node_name, summary, restored_context, state.compact_tail)
    will_retrigger_next_turn = post_compact_tokens >= _COMPACT_TRIGGER_TOKENS
    if will_retrigger_next_turn:
        _record_compact_failure(
            state,
            "post_compact_over_threshold",
            f"post_compact_tokens={post_compact_tokens}; threshold={_COMPACT_TRIGGER_TOKENS}",
        )
        state.compact_failure_count += 1
        state.needs_compact = False
        return state

    new_compacted_count = state.compacted_count + len(state.compact_prefix)
    boundary_record = {
        "type": "context_compact_boundary",
        "created_at": _utc_now(),
        "compacted_count": new_compacted_count,
        "trigger": "auto",
        "pre_compact_tokens": state.total_tokens,
        "post_compact_tokens": post_compact_tokens,
        "compacted_record_count": len(state.compact_prefix),
        "preserved_start_index": len(state.compact_prefix),
        "preserved_record_count": len(state.compact_tail),
        "compact_failure_count_before": state.compact_failure_count,
        "autocompact_threshold_tokens": _COMPACT_TRIGGER_TOKENS,
        "tool_history_cleared_results": state.tool_history_diagnostics.cleared_results,
        "tool_history_tokens_saved": state.tool_history_diagnostics.tokens_saved,
        "will_retrigger_next_turn": will_retrigger_next_turn,
    }
    summary_record = {
        "type": "context_summary",
        "created_at": _utc_now(),
        "summary": summary,
        "restored_context": restored_context,
    }
    append_items(state.run_id, state.node_name, [boundary_record, summary_record])

    state.compressed_context = summary
    state.working_context = list(state.compact_tail)
    state.restored_context = restored_context
    state.compacted_count = new_compacted_count
    state.records = read_records(state.run_id, state.node_name)
    state.compact_failure_count = 0
    state.estimated_tokens = _estimate_projected_tokens(state)
    return state


def filter_records_for_model_context(records: list[dict]) -> list[dict]:
    return sanitize_records_for_context(records)


def truncate_context_turn(records: list[dict]) -> dict:
    return _render_truncated_turn_note(records)


def assemble_model_input_items(state: ContextState, *, include_todo_context: bool = True) -> list[dict]:
    """输入: 一个完整的 `ContextState`, 里面已经包含固定区、压缩区、工作区。
    处理: 把这三个区投影成真正能送给 Responses API 的 `input_items` 列表。
    输出: `list[dict]`, 也就是本轮最终要发给模型的上下文输入。"""
    items: list[dict] = list(state.fixed_context)

    if state.compressed_context is not None:
        items.append(
            {
                "role": "user",
                "content": _render_summary_for_context(state.node_name, state.compressed_context),
            }
        )

    items.extend(project_records_to_model_input_items(state.restored_context))
    items.extend(project_records_to_model_input_items(state.working_context))
    if include_todo_context and state.todo_context is not None:
        items.append(state.todo_context)
    return items


def run_context_pipeline(run_id: str, node_name: str) -> ContextState:
    """输入: `run_id`/`node_name`, 这是 runtime 传给上下文层的最小定位参数。
    处理: 顺序执行 build -> trim -> measure -> should_compact -> run_compaction 这条完整 context pipeline。
    输出: 一个最终 `ContextState`, 里面已经是当前可直接投影给模型的固定区/压缩区/工作区状态。"""
    state = build_context(run_id, node_name)
    state = trim_tool_results(state)
    state = measure_context_budget(state)
    should_compact(state)
    state = run_compaction(state)
    return state


def build_openai_input_items(run_id: str, node_name: str, *, include_todo_context: bool = True) -> list[dict]:
    """输入: `run_id`/`node_name`, 这是 runtime 调这层时给的最小定位信息。
    处理: 调完整的 `run_context_pipeline(...)`, 然后把内部状态投影成最终 API `input_items`。
    输出: 一份最终 `list[dict]` 上下文输入; 是否该压缩会留在内部状态判断里。"""
    state = run_context_pipeline(run_id, node_name)
    return assemble_model_input_items(state, include_todo_context=include_todo_context)


def split_records_for_compaction(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """输入: 一串工作区 `records` 列表。
    处理: 从后往前保留最近的一段 tail, 剩下更早的 prefix 作为待压缩区域。
    输出: 一个二元组 `(prefix, tail)`, 前者准备拿去 summary, 后者继续保留原始细节。"""
    return split_for_compaction(records, DEFAULT_CONTEXT_POLICY)


def _count_recent_compact_failures(records: list[dict]) -> int:
    failures = 0
    for record in reversed(records):
        record_type = record.get("type")
        if record_type in {"context_compact_boundary", "context_summary"}:
            break
        if record_type == "context_compact_failure":
            failures += 1
    return failures


def _rebalance_tool_pairs(prefix: list[dict], tail: list[dict]) -> tuple[list[dict], list[dict]]:
    """输入: 一次初步切分后的 `(prefix, tail)` 两段 records。
    处理: 检查切分点是否把 `function_call` / `function_call_output` 这类工具配对拆开; 如果拆开, 就把未闭合的调用移动到 tail。
    输出: 一个重新平衡后的 `(prefix, tail)`, 保证 prefix 单独送去 compact 时不会含有缺失 output 的 tool call。"""
    if not prefix or not tail:
        return prefix, tail

    prefix_output_ids = {
        record.get("call_id")
        for record in prefix
        if record.get("type") == "function_call_output" and record.get("call_id")
    }
    unmatched_prefix_call_ids = {
        record.get("call_id")
        for record in prefix
        if record.get("type") == "function_call"
        and record.get("call_id")
        and record.get("call_id") not in prefix_output_ids
    }

    if not unmatched_prefix_call_ids:
        return prefix, tail

    moved_calls: list[dict] = []
    kept_prefix: list[dict] = []
    for record in prefix:
        if record.get("type") == "function_call" and record.get("call_id") in unmatched_prefix_call_ids:
            moved_calls.append(record)
        else:
            kept_prefix.append(record)

    return kept_prefix, moved_calls + tail


def build_compaction_input_items(state: ContextState, prefix: list[dict]) -> list[dict]:
    """输入: 一个 `ContextState` 和一段准备被压缩的 `prefix records`。
    处理: 把固定区、已有压缩区、这次待压缩前缀拼成一次 summary invoke 的输入源。
    输出: `list[dict]`, 也就是专门给 compact 模型看的输入 items。"""
    items: list[dict] = list(state.fixed_context)
    if state.compressed_context is not None:
        items.append(
            {
                "role": "user",
                "content": _render_summary_for_context(state.node_name, state.compressed_context),
            }
        )
    items.extend(project_records_to_model_input_items(prefix))
    return items


def project_records_to_model_input_items(records: list[dict]) -> list[dict]:
    """输入: 一组本地标准化 records, 可能混着 user/assistant/tool 记录。
    处理: 逐条把 record 映射成 OpenAI Responses API 能接受的 item 结构。
    输出: `list[dict]`, 只保留成功投影出来的 items。"""
    items: list[dict] = []
    for record in records:
        item = record_to_model_input_item(record)
        if item is not None:
            items.append(item)
    return items


def select_recent_context_turns(records: list[dict]) -> list[list[dict]]:
    turns: list[list[dict]] = []
    current_tool_turn: list[dict] = []
    open_tool_call_ids: set[str] = set()

    def flush_tool_turn() -> None:
        nonlocal current_tool_turn, open_tool_call_ids
        if current_tool_turn:
            turns.append(current_tool_turn)
            current_tool_turn = []
            open_tool_call_ids = set()

    for record in records:
        if record.get("type") in {"context_compact_boundary", "context_summary", "context_compact_failure", "token_usage"}:
            continue

        if record.get("type") == "function_call":
            if current_tool_turn and not open_tool_call_ids:
                flush_tool_turn()
            call_id = record.get("call_id")
            if isinstance(call_id, str) and call_id:
                open_tool_call_ids.add(call_id)
            current_tool_turn.append(record)
            continue

        if record.get("type") == "function_call_output":
            current_tool_turn.append(record)
            call_id = record.get("call_id")
            if isinstance(call_id, str):
                open_tool_call_ids.discard(call_id)
            continue

        flush_tool_turn()
        turns.append([record])

    flush_tool_turn()
    return turns


def project_turn_records_to_model_input_items(records: list[dict]) -> list[dict]:
    items: list[dict] = []
    note_call_ids: set[str] = set()

    for record in records:
        if record.get("type") == "function_call" and _function_call_needs_context_note(record):
            call_id = record.get("call_id")
            if isinstance(call_id, str):
                note_call_ids.add(call_id)
            items.append(_render_function_call_note(record))
            continue

        if record.get("type") == "function_call_output" and record.get("call_id") in note_call_ids:
            items.append(_render_function_output_note(record))
            continue

        item = record_to_model_input_item(record)
        if item is not None:
            items.append(item)
    return items


def record_to_model_input_item(record: dict) -> dict | None:
    """输入: 一条本地标准 record 字典。
    处理: 按 record 的类型判断它应该投影成 user/assistant 消息, 还是 function_call / function_call_output。
    输出: 一个 API input item 字典; 如果这条 record 不该进模型上下文, 就返回 `None`。"""
    role = record.get("role")
    item_type = record.get("type")

    if role in {"user", "assistant"}:
        content = record.get("content")
        if not content:
            return None
        return {"role": role, "content": _clean_context_text(str(content))}

    if item_type == "function_call":
        call_id = record.get("call_id")
        name = record.get("name")
        if not (call_id and name):
            return None
        item = {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": _clean_context_text(str(record.get("arguments", ""))),
        }
        if record.get("id"):
            item["id"] = record["id"]
        thinking = record.get("thinking")
        if isinstance(thinking, str) and thinking:
            item["thinking"] = _clean_context_text(thinking)
        return item

    if item_type == "function_call_output":
        call_id = record.get("call_id")
        if not call_id:
            return None
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": _clean_context_text(str(record.get("output", ""))),
        }

    return None


def _function_call_needs_context_note(record: dict) -> bool:
    arguments = str(record.get("arguments", ""))
    if _has_internal_context_marker(arguments):
        return True
    return False


def _render_function_call_note(record: dict) -> dict:
    name = str(record.get("name") or "unknown_tool")
    arguments = _clean_context_text(str(record.get("arguments", "")))
    preview = build_range_preview(arguments, 400)
    content = "\n".join(
        [
            f"历史工具调用摘要：{name}",
            preview,
        ]
    )
    return {"role": "user", "content": content}


def _render_function_output_note(record: dict) -> dict:
    output = _clean_context_text(str(record.get("output", "")))
    preview = build_range_preview(output, 400)
    content = "\n".join(
        [
            "历史工具结果摘要：",
            preview,
        ]
    )
    return {"role": "user", "content": content}


def _render_truncated_turn_note(records: list[dict]) -> dict:
    text = _clean_context_text(_render_records_for_truncated_note(records))
    preview = build_range_preview(text, 400)
    content = "\n".join(
        [
            "历史轮次摘要：",
            preview,
        ]
    )
    return {"role": "user", "content": content}


def _render_records_for_truncated_note(records: list[dict]) -> str:
    parts: list[str] = []
    for record in records:
        role = record.get("role")
        record_type = record.get("type")
        if role in {"user", "assistant"}:
            parts.append(f"{role}: {record.get('content', '')}")
        elif record_type == "function_call":
            parts.append(f"tool_call {record.get('name')}: {record.get('arguments', '')}")
        elif record_type == "function_call_output":
            parts.append(f"tool_result: {record.get('output', '')}")
        else:
            parts.append(json.dumps(record, ensure_ascii=False))
    return "\n".join(parts)


def _has_internal_context_marker(text: str) -> bool:
    return any(marker in text for marker in _INTERNAL_CONTEXT_MARKERS)


def clean_model_visible_text(text: str) -> str:
    return _clean_context_text(text)


def _clean_context_text(text: str) -> str:
    cleaned = text
    replacements = {
        "_snipped_for_context": "历史内容范围预览",
        "_omitted": "历史内容范围预览",
        "[snipped for context]": "[历史内容范围预览]",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    max_chars = max(1, max_tokens * _CHARS_PER_ESTIMATED_TOKEN)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _estimate_projected_tokens(state: ContextState) -> int:
    """输入: 一个 `ContextState`, 默认关注压缩区和工作区这两块真实会进模型的内容。
    处理: 把这些区投影成文本后做轻量 token 估算, 用于老逻辑或兼容场景。
    输出: 一个 `int`, 表示当前 projected context 的总估算 token。"""
    total = _estimate_records_tokens(state.working_context)
    if state.compressed_context is not None:
        total += _estimate_tokens(_render_summary_for_context(state.node_name, state.compressed_context))
    total += _estimate_records_tokens(state.restored_context)
    if state.todo_context is not None:
        total += _estimate_record_tokens(state.todo_context)
    return total


def _estimate_post_compact_tokens(
    node_name: str,
    summary: dict,
    restored_context: list[dict],
    preserved_records: list[dict],
) -> int:
    return (
        _estimate_tokens(_render_summary_for_context(node_name, summary))
        + _estimate_records_tokens(restored_context)
        + _estimate_records_tokens(preserved_records)
    )


def _build_restored_context_notes(compacted_prefix: list[dict], preserved_records: list[dict]) -> list[dict]:
    """Build a short post-compact note naming relevant files without injecting file contents."""
    prefix_paths = _extract_recent_file_paths(compacted_prefix)
    if not prefix_paths:
        return []

    preserved_paths = set(_extract_recent_file_paths(preserved_records))
    restored_paths = [path for path in prefix_paths if path not in preserved_paths]
    if not restored_paths:
        return []

    restored_paths = restored_paths[:_MAX_RESTORED_CONTEXT_FILES]
    content = "最近相关文件：\n" + "\n".join(f"- {path}" for path in restored_paths)
    return [{"role": "user", "content": content}]


def _extract_recent_file_paths(records: list[dict]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for record in reversed(records):
        if record.get("type") != "function_call":
            continue
        for path in _paths_from_tool_arguments(record.get("arguments", "")):
            normalized = _normalize_path_for_context(path)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            paths.append(normalized)
            if len(paths) >= _MAX_RESTORED_CONTEXT_FILES:
                return paths
    return paths


def _paths_from_tool_arguments(arguments: Any) -> list[str]:
    if isinstance(arguments, str):
        try:
            payload = json.loads(arguments)
        except json.JSONDecodeError:
            return _paths_from_text(arguments)
    elif isinstance(arguments, dict):
        payload = arguments
    else:
        return []

    paths: list[str] = []
    for key in ("path", "file_path", "filepath", "target_path", "workdir"):
        value = payload.get(key)
        if isinstance(value, str):
            paths.append(value)
    for key in ("paths", "file_paths", "files"):
        value = payload.get(key)
        if isinstance(value, list):
            paths.extend(item for item in value if isinstance(item, str))
    return paths


def _paths_from_text(text: str) -> list[str]:
    candidates = re.findall(r"(?:(?:[A-Za-z]:)?[\\/])?[\w .@()+-]+(?:[\\/][\w .@()+-]+)+", text)
    return [candidate.strip().strip("\"'") for candidate in candidates]


def _normalize_path_for_context(path: str) -> str:
    cleaned = path.strip().strip("\"'")
    if not cleaned:
        return ""
    return cleaned.replace("\\", "/")


def _estimate_records_tokens(records: list[dict]) -> int:
    """输入: 一组 records 列表。
    处理: 逐条读取每条 record 的 `estimated_tokens`, 没有的话再现场估算并求和。
    输出: 一个 `int`, 表示这批 records 的总估算 token。"""
    return sum(_estimate_record_tokens(record) for record in records)


def _estimate_record_tokens(record: dict) -> int:
    """输入: 一条 record 字典, 可能已经带 `estimated_tokens` 字段。
    处理: 优先用缓存好的 token 元数据, 没有的话再根据内容类型做一次临时估算。
    输出: 一个 `int`, 表示这条 record 的估算 token 数。"""
    cached = record.get("estimated_tokens")
    if isinstance(cached, int) and cached >= 0:
        return cached

    role = record.get("role")
    item_type = record.get("type")

    if role in {"user", "assistant"}:
        return _estimate_tokens(record.get("content", ""))

    if item_type == "function_call":
        return _estimate_tokens(record.get("name", "")) + _estimate_tokens(record.get("arguments", ""))

    if item_type == "function_call_output":
        return _estimate_tokens(record.get("output", ""))

    if item_type == "context_summary":
        return _estimate_tokens(record.get("summary", {}))

    return _estimate_tokens(record)


def _with_estimated_tokens(record: dict) -> dict:
    """输入: 一条从历史里读出来的 record。
    处理: 如果它还没有 `estimated_tokens`, 就补算一个并塞回这条记录。
    输出: 一条可直接用于预算统计的 record 字典。"""
    if isinstance(record.get("estimated_tokens"), int):
        return record
    cloned = dict(record)
    cloned["estimated_tokens"] = _estimate_record_tokens(cloned)
    return cloned


def _estimate_tokens(value: Any) -> int:
    """输入: 任意可转文本的值, 比如字符串、字典、列表。
    处理: 统一转成文本后按字符数粗略估算 token, 作为轻量预算手段。
    输出: 一个 `int` 估算值。"""
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
    return max(1, (len(text) + _CHARS_PER_ESTIMATED_TOKEN - 1) // _CHARS_PER_ESTIMATED_TOKEN)


def _extract_response_text(response: Any) -> str:
    """输入: 一次 Responses API 的原始响应对象。
    处理: 优先走 `output_text`, 不行再从 `output[].content[].text` 里把文本拼出来。
    输出: 一个纯文本字符串, 方便后面解析 compact summary。"""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in output:
        content = getattr(item, "content", None) or []
        for block in content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts).strip()


def _parse_structured_summary(summary_text: str) -> dict:
    """输入: 一段 compact 模型返回的原始 summary 文本。
    处理: 去掉 `<analysis>`, 提取 `<summary>`, 再按 1 到 9 段结构解析成字段字典。
    输出: 一个 `dict`, 键就是我们定义的九段结构化 summary 字段。"""
    text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", summary_text, flags=re.IGNORECASE).strip()
    match = re.search(r"<summary>([\s\S]*?)</summary>", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1).strip()

    sections = {
        "primary_request_and_intent": [],
        "key_technical_concepts": [],
        "files_and_code_sections": [],
        "errors_and_fixes": [],
        "problem_solving": [],
        "all_user_messages": [],
        "pending_tasks": [],
        "current_work": "",
        "optional_next_step": "",
    }

    mapping = {
        "1": "primary_request_and_intent",
        "2": "key_technical_concepts",
        "3": "files_and_code_sections",
        "4": "errors_and_fixes",
        "5": "problem_solving",
        "6": "all_user_messages",
        "7": "pending_tasks",
        "8": "current_work",
        "9": "optional_next_step",
    }
    positions = list(re.finditer(r"(?m)^\s*(\d+)\.\s+[^\n]*\n?", text))
    for idx, pos in enumerate(positions):
        section_num = pos.group(1)
        key = mapping.get(section_num)
        if key is None:
            continue
        start = pos.end()
        end = positions[idx + 1].start() if idx + 1 < len(positions) else len(text)
        body = text[start:end].strip()
        if key in {"current_work", "optional_next_step"}:
            sections[key] = body
        else:
            sections[key] = _split_section_lines(body)
    return sections


def _split_section_lines(body: str) -> list[str]:
    """输入: summary 某一段的正文字符串。
    处理: 按行拆开, 去掉空行和前面的 `-`/`*` 这种列表符号。
    输出: 一个 `list[str]`, 适合作为结构化 summary 的数组字段。"""
    lines = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s*", "", line)
        lines.append(line)
    return lines


def _render_summary_for_context(node_name: str, summary: dict) -> str:
    """输入: `node_name` 和一份结构化 `summary dict`。
    处理: 把结构化摘要重新渲染成一条给主模型看的连续文本 summary message。
    输出: 一个 `str`, 会作为压缩区进入最终 context。"""
    del node_name
    lines = [
        "这轮会话是在此前上下文已压缩的基础上继续进行。",
        "下面的摘要覆盖的是更早一段对话内容。",
        "",
        "摘要：",
    ]

    section_titles = [
        ("primary_request_and_intent", "1. 主要需求与意图", None),
        ("key_technical_concepts", "2. 关键技术概念", None),
        ("files_and_code_sections", "3. 文件与代码片段", None),
        ("errors_and_fixes", "4. 错误与修复", None),
        ("problem_solving", "5. 问题解决过程", None),
        ("all_user_messages", "6. 所有用户消息", "task_inputs_and_feedback"),
        ("pending_tasks", "7. 待办事项", None),
        ("current_work", "8. 当前工作", "current_execution_state"),
        ("optional_next_step", "9. 可选下一步", None),
    ]

    for key, title, fallback_key in section_titles:
        value = summary.get(key)
        if not value and fallback_key:
            value = summary.get(fallback_key)
        if not value:
            continue
        lines.append("")
        lines.append(title + ":")
        if isinstance(value, list):
            for item in value:
                lines.append(f"- {item}")
        else:
            lines.append(str(value))

    lines.append("")
    lines.append("最近记录保持原样保留。")
    lines.append("请直接从上次停下的位置继续，不要重复总结。")
    return "\n".join(lines)


def _utc_now() -> str:
    """输入: 没有输入参数。
    处理: 读取当前 UTC 时间并格式化成简单的 ISO 字符串。
    输出: 一个 `str`, 形如 `2026-05-02T10:20:30Z`。"""
    return datetime.now(UTC).replace(tzinfo=None).isoformat(timespec="seconds") + "Z"


__all__ = [
    "ContextState",
    "build_context",
    "trim_tool_results",
    "measure_context_budget",
    "should_compact",
    "run_compaction",
    "invoke_compact_summary",
    "compact_context",
    "assemble_model_input_items",
    "filter_records_for_model_context",
    "clean_model_visible_text",
    "select_recent_context_turns",
    "truncate_context_turn",
    "record_to_model_input_item",
    "project_records_to_model_input_items",
    "project_context",
    "run_context_pipeline",
    "build_openai_input_items",
]


project_context = assemble_model_input_items
