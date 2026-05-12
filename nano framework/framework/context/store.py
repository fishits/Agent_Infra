"""Storage-first memory for Responses API sessions.

The local memory files are optimized for:
1. readable debugging history
2. preserving useful response details
3. keeping context assembly in a separate layer
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..mem_paths import MEM_ROOT as _DEFAULT_MEM_ROOT, run_dir

_MEM_ROOT = _DEFAULT_MEM_ROOT
_CHARS_PER_ESTIMATED_TOKEN = 4


def append_items(run_id: str, node_name: str, raw_items: list[Any]) -> None:
    """输入: `run_id`/`node_name` 用来定位会话文件, `raw_items` 是一批原始事件对象列表。
    处理: 逐条转成 dict, 做存储层轻过滤和字段裁剪, 然后追加写回 `.mem/<run_id>/<node>.json`。
    输出: 没有返回值; 结果是本地历史文件里多了新的标准化 records。"""
    if not raw_items:
        return

    records = read_records(run_id, node_name)

    for item in raw_items:
        raw = _to_dict(item)
        if raw is None:
            continue
        record = _prune_for_storage(raw)
        if record is None:
            continue

        records.append(record)

    _write_json(_records_path(run_id, node_name), records)


def clear_records(run_id: str, node_name: str) -> None:
    """Clear one node's local conversation records while leaving durable workspace state untouched."""
    _write_json(_records_path(run_id, node_name), [])
    clear_response_state(run_id, node_name)


def load_response_state(run_id: str, node_name: str) -> dict:
    """输入: `run_id`/`node_name`, 用来找到这个 node 对应的 `.state.json` 文件。
    处理: 读取并解析状态文件, 只关心像 `previous_response_id` 这种节点级会话状态。
    输出: 一个 `dict`, 形如 `{\"previous_response_id\": str | None}`。"""
    path = _state_path(run_id, node_name)
    if not path.exists():
        return {"previous_response_id": None}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"previous_response_id": None}

    if not isinstance(data, dict):
        return {"previous_response_id": None}

    previous_response_id = data.get("previous_response_id")
    if previous_response_id is not None and not isinstance(previous_response_id, str):
        previous_response_id = None

    return {"previous_response_id": previous_response_id}


def save_response_state(run_id: str, node_name: str, previous_response_id: str | None) -> None:
    """输入: `run_id`/`node_name` 和一个 `previous_response_id` 字符串或 `None`。
    处理: 把这份轻量状态包装成字典并落盘到 node 对应的 `.state.json`。
    输出: 没有返回值; 结果是本地状态文件被创建或覆盖。"""
    payload = {"previous_response_id": previous_response_id}
    _write_json(_state_path(run_id, node_name), payload)


def clear_response_state(run_id: str, node_name: str) -> None:
    """输入: `run_id`/`node_name`, 指向某个 node 的状态文件。
    处理: 复用 `save_response_state(...)`, 把 `previous_response_id` 置空。
    输出: 没有返回值; 结果是这个 node 的会话续接状态被清空。"""
    save_response_state(run_id, node_name, None)


def read_latest_compaction(run_id: str, node_name: str) -> dict:
    """输入: `run_id`/`node_name`, 用来读取某个 node 的完整本地历史 records。
    处理: 从后往前找最近一次 `context_compact_boundary + context_summary` 这一对压缩标记。
    输出: 一个 `dict`, 包含 `boundary_index`、`compacted_count`、`summary`、`working_start_index` 等信息。"""
    records = read_records(run_id, node_name)
    latest: dict | None = None

    for index, record in enumerate(records):
        if record.get("type") != "context_compact_boundary":
            continue
        summary_index = index + 1
        if summary_index >= len(records):
            continue
        summary_record = records[summary_index]
        if summary_record.get("type") != "context_summary":
            continue
        latest = {
            "boundary_index": index,
            "summary_index": summary_index,
            "compacted_count": record.get("compacted_count", 0),
            "created_at": record.get("created_at"),
            "summary": summary_record.get("summary"),
            "restored_context": summary_record.get("restored_context", []),
            "boundary": record,
        }
        latest["working_start_index"] = latest["compacted_count"]

    return latest or {
        "boundary_index": -1,
        "summary_index": -1,
        "working_start_index": 0,
        "compacted_count": 0,
        "created_at": None,
        "summary": None,
    }


def read_latest_token_usage(run_id: str, node_name: str) -> dict:
    """输入: `run_id`/`node_name`, 用来读取某个 node 的完整本地历史 records。
    处理: 从后往前找最近一条 `token_usage` 记录, 作为这个 node 最近一次真实 API 用量快照。
    输出: 一个 `dict`, 形如 `{\"input_tokens\": int | None, \"output_tokens\": int | None, \"total_tokens\": int | None}`。"""
    records = read_records(run_id, node_name)
    for record in reversed(records):
        if record.get("type") != "token_usage":
            continue
        return {
            "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"),
            "total_tokens": record.get("total_tokens"),
            "ttft": record.get("ttft"),
            "total_time": record.get("total_time"),
            "decode_tokens_per_second": record.get("decode_tokens_per_second"),
            "cached_tokens": record.get("cached_tokens"),
            "cache_text": record.get("cache_text"),
            "created_at": record.get("created_at"),
            "response_id": record.get("response_id"),
        }
    return {
        "input_tokens": None,
        "output_tokens": None,
        "total_tokens": None,
        "ttft": None,
        "total_time": None,
        "decode_tokens_per_second": None,
        "cached_tokens": None,
        "cache_text": None,
        "created_at": None,
        "response_id": None,
    }


def _prune_for_storage(raw: dict) -> dict | None:
    """输入: 一条原始事件 `dict`, 可能是 user/assistant/tool/reasoning/compact 记录。
    处理: 按类型挑出对本地历史真正有用的字段, 丢掉噪声, 顺手补 `estimated_tokens` 等元数据。
    输出: 一个更干净的标准 record 字典; 如果这条记录不值得存, 就返回 `None`。"""
    role = raw.get("role")
    item_type = raw.get("type")

    if role in {"user", "assistant"}:
        content = raw.get("content")
        if not content:
            return None
        record = {
            "role": role,
            "content": content,
        }
        thinking = raw.get("thinking")
        if isinstance(thinking, str) and thinking:
            record["thinking"] = thinking
        record["estimated_tokens"] = _estimate_tokens(content)
        return record

    if item_type == "reasoning":
        summary = raw.get("summary") or []
        if not summary:
            return None
        record = {
            "type": "reasoning",
            "summary": summary,
        }
        if raw.get("id"):
            record["id"] = raw["id"]
        if raw.get("status"):
            record["status"] = raw["status"]
        record["estimated_tokens"] = _estimate_tokens(summary)
        return record

    if item_type == "function_call":
        call_id = raw.get("call_id")
        name = raw.get("name")
        if not (call_id and name):
            return None
        record = {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": _sanitize_function_call_arguments(name, raw.get("arguments", "")),
        }
        if raw.get("id"):
            record["id"] = raw["id"]
        if raw.get("status"):
            record["status"] = raw["status"]
        thinking = raw.get("thinking")
        if isinstance(thinking, str) and thinking:
            record["thinking"] = thinking
        record["estimated_tokens"] = _estimate_tokens(name) + _estimate_tokens(record["arguments"])
        return record

    if item_type == "function_call_output":
        call_id = raw.get("call_id")
        if not call_id:
            return None
        output = raw.get("output", "")
        return {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
            "estimated_tokens": _estimate_tokens(output),
        }

    if item_type == "context_compact_boundary":
        record = {"type": "context_compact_boundary"}
        if raw.get("created_at"):
            record["created_at"] = raw["created_at"]
        for key in (
            "compacted_count",
            "pre_compact_tokens",
            "post_compact_tokens",
            "compacted_record_count",
            "preserved_start_index",
            "preserved_record_count",
            "compact_failure_count_before",
            "autocompact_threshold_tokens",
            "tool_history_cleared_results",
            "tool_history_tokens_saved",
        ):
            if isinstance(raw.get(key), int):
                record[key] = raw[key]
        if raw.get("trigger"):
            record["trigger"] = raw["trigger"]
        if isinstance(raw.get("will_retrigger_next_turn"), bool):
            record["will_retrigger_next_turn"] = raw["will_retrigger_next_turn"]
        if raw.get("id"):
            record["id"] = raw["id"]
        record["estimated_tokens"] = 1
        return record

    if item_type == "context_summary":
        summary = raw.get("summary")
        if not isinstance(summary, dict):
            return None
        record = {
            "type": "context_summary",
            "summary": summary,
        }
        restored_context = raw.get("restored_context")
        if isinstance(restored_context, list):
            record["restored_context"] = restored_context
        if raw.get("created_at"):
            record["created_at"] = raw["created_at"]
        if raw.get("id"):
            record["id"] = raw["id"]
        record["estimated_tokens"] = _estimate_tokens(summary)
        return record

    if item_type == "token_usage":
        input_tokens = raw.get("input_tokens")
        output_tokens = raw.get("output_tokens")
        total_tokens = raw.get("total_tokens")
        if not all(isinstance(value, int) for value in (input_tokens, output_tokens, total_tokens)):
            return None
        record = {
            "type": "token_usage",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_tokens": 1,
        }
        if raw.get("created_at"):
            record["created_at"] = raw["created_at"]
        if raw.get("response_id"):
            record["response_id"] = raw["response_id"]
        ttft = raw.get("ttft")
        total_time = raw.get("total_time")
        decode_tokens_per_second = raw.get("decode_tokens_per_second")
        cached_tokens = raw.get("cached_tokens")
        cache_text = raw.get("cache_text")
        if isinstance(ttft, (int, float)):
            record["ttft"] = float(ttft)
        if isinstance(total_time, (int, float)):
            record["total_time"] = float(total_time)
        if isinstance(decode_tokens_per_second, (int, float)):
            record["decode_tokens_per_second"] = float(decode_tokens_per_second)
        if isinstance(cached_tokens, int):
            record["cached_tokens"] = cached_tokens
        if isinstance(cache_text, str) and cache_text:
            record["cache_text"] = cache_text
        return record

    return None


def _to_dict(item: Any) -> dict | None:
    """输入: 任意事件对象, 可能本来就是 `dict`, 也可能是 SDK 返回对象。
    处理: 优先尝试 `model_dump()` 或 `to_dict()` 把对象转成普通字典。
    输出: 一个 `dict`; 如果这个对象没法可靠转字典, 就返回 `None`。"""
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "to_dict"):
        return item.to_dict()
    return None


def _sanitize_function_call_arguments(tool_name: str, arguments: str) -> str:
    """落盘时保留原始参数，展示层和上下文层再各自做范围式裁切。"""
    del tool_name
    return arguments if isinstance(arguments, str) else ""


def read_records(run_id: str, node_name: str) -> list[dict]:
    """输入: `run_id`/`node_name`, 对应某个 node 的历史 records 文件。
    处理: 从磁盘读取 JSON, 解析并校验它是不是一个列表。
    输出: `list[dict]`; 如果文件不存在或损坏, 返回空列表。"""
    path = _records_path(run_id, node_name)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    return data if isinstance(data, list) else []


def _estimate_tokens(value: Any) -> int:
    """输入: 任意可序列化值, 比如字符串、字典、列表。
    处理: 先转成文本, 再用字符数近似估算 token 数, 作为轻量预算元数据。
    输出: 一个 `int` 估算值, 至少为 1 或 0。"""
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


def _write_json(path: Path, data: Any) -> None:
    """输入: 一个文件路径 `Path` 和任意要保存的数据对象。
    处理: 自动创建父目录, 然后把数据格式化成 UTF-8 JSON 写入磁盘。
    输出: 没有返回值; 结果是目标路径上的 JSON 文件被写好。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _records_path(run_id: str, node_name: str) -> Path:
    """输入: `run_id` 和原始 `node_name` 字符串。
    处理: 把 node 名里不安全的路径分隔符替换掉, 再拼出 records 文件路径。
    输出: 一个 `Path`, 指向 `.mem/<run_id>/<safe_node>.json`。"""
    safe = node_name.replace("/", "_").replace("\\", "_")
    return run_dir(run_id, _MEM_ROOT) / f"{safe}.json"


def _state_path(run_id: str, node_name: str) -> Path:
    """输入: `run_id` 和原始 `node_name` 字符串。
    处理: 和 records 路径一样先做安全化, 但目标是状态文件而不是历史文件。
    输出: 一个 `Path`, 指向 `.mem/<run_id>/<safe_node>.state.json`。"""
    safe = node_name.replace("/", "_").replace("\\", "_")
    return run_dir(run_id, _MEM_ROOT) / f"{safe}.state.json"


__all__ = [
    "append_items",
    "clear_records",
    "clear_response_state",
    "load_response_state",
    "read_latest_compaction",
    "read_latest_token_usage",
    "read_records",
    "save_response_state",
]
