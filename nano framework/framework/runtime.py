"""图运行时。"""

from __future__ import annotations

import re
import signal
import shutil
from typing import Any, Callable, Iterator

from .agent import node_invoke
from .mem import clear_records
from .range_preview import build_range_preview, extract_preview_ranges


_thinking_open = False
_pending_tool_hint: dict | None = None
_TOOL_RESULT_PREVIEW_CHARS = 4000
_ROUTE_PREVIEW_CHARS = 400
_NODE_MESSAGE_PREVIEW_CHARS = 500

_ARG_SNIFF_RE = re.compile(
    r'"(?:path|paths|command|query|log_path)"\s*:\s*"([^"]{1,120})',
)
_CONTROL_TOOLS = {"chat", "decide", "new_decide"}


def run(
    graph: dict,
    user_message: str = "",
    stream: bool = False,
    event_output: Callable[[dict], None] | None = None,
):
    """Run one graph object."""
    if stream:
        events = _run_stream(graph, user_message)
        output = event_output or _default_event_output
        for event in events:
            output(event)
        return graph["state"]

    interrupted = False

    def signal_handler(signum, frame):
        del signum, frame
        nonlocal interrupted
        interrupted = True
        print("\n\n[警告] 收到中断信号，正在停止……")

    signal.signal(signal.SIGINT, signal_handler)
    try:
        output = event_output or _default_event_output
        for event in _run_stream(graph, user_message):
            if interrupted:
                print("[停止] 执行已中断")
                break
            output(event)
    finally:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    return graph["state"]


def _run_stream(
    graph: dict,
    user_message: str,
) -> Iterator[dict]:
    """Yield graph events while scheduling one node at a time."""
    pending_message = user_message
    if user_message and not graph["state"].get("root_user_message"):
        graph["state"]["root_user_message"] = user_message
    node_index = {node["name"]: node for node in graph["nodes"]}

    while graph["state"]["current_node"] != "end":
        current_node = graph["state"]["current_node"]
        node = node_index.get(current_node)
        if node is None:
            yield {"type": "error", "message": f"节点不存在：{current_node}"}
            return

        next_node = None
        clear_target_memory = False
        for event in node_invoke(graph, node, pending_message):
            yield event
            if event["type"] == "chat_request":
                pending_message = _default_user_input(event["message"])
                break
            if event["type"] == "next_node":
                next_node = event["node"]
                clear_target_memory = bool(event.get("clear_target_memory"))
        else:
            pending_message = ""
            if next_node is None:
                continue
            target, content = next_node
            if target == "end":
                graph["state"]["current_node"] = "end"
            else:
                if clear_target_memory:
                    clear_records(graph["state"]["run_id"], target)
                yield {
                    "type": "node_transition",
                    "from": current_node,
                    "to": target,
                    "from_display": _node_display_name(node),
                    "to_display": _node_display_name(node_index.get(target), fallback=target),
                    "content_preview": _route_preview_text(content),
                }
                graph["state"]["current_node"] = target
                pending_message = content
            continue

        pending_message = pending_message or ""

    yield {"type": "completed"}


def _default_user_input(message: str) -> str:
    """Read user input for chat() when running interactively."""
    global _thinking_open
    if _thinking_open:
        print()
        _thinking_open = False
    _print_block("对话", message)
    return input("[用户] ").strip() or "(无输入)"


def _default_event_output(event: dict) -> None:
    """Render one runtime event for CLI use."""
    global _thinking_open, _pending_tool_hint
    event_type = event.get("type")

    if event_type == "reasoning":
        content = event.get("content", "")
        if content:
            if not _thinking_open:
                _flush_tool_hint()
                print()
                print("[思考] ", end="", flush=True)
                _thinking_open = True
            print(content, end="", flush=True)

    elif event_type == "node_start":
        _flush_tool_hint()
        if _thinking_open:
            print()
            _thinking_open = False
        node_display = event.get("display_name", event.get("node", "未知"))
        message = (event.get("message") or "").strip()
        root_message = (event.get("root_message") or "").strip()
        body = node_display
        if root_message:
            root_preview = _route_preview_text(root_message)
            if len(root_preview) > _NODE_MESSAGE_PREVIEW_CHARS:
                root_preview = root_preview[:_NODE_MESSAGE_PREVIEW_CHARS]
            body += f"\n[需求] {root_preview}"
        if message:
            preview = _route_preview_text(message)
            if len(preview) > _NODE_MESSAGE_PREVIEW_CHARS:
                preview = preview[:_NODE_MESSAGE_PREVIEW_CHARS]
            body += f"\n[输入] {preview}"
        else:
            body += "\n[输入] （从上下文继续）"
        _print_block("节点", body)

    elif event_type == "tool_call_start":
        if _thinking_open:
            print()
            _thinking_open = False
        name = event.get("name", "")
        if name not in _CONTROL_TOOLS:
            _pending_tool_hint = {"name": name, "printed_arg": False}
            print(f"\n[调用] {name}", end="", flush=True)

    elif event_type == "tool_call_args":
        name = event.get("name", "")
        if name in _CONTROL_TOOLS:
            pass
        elif _pending_tool_hint and not _pending_tool_hint["printed_arg"]:
            delta = event.get("delta", "")
            if delta:
                _pending_tool_hint.setdefault("_buf", "")
                _pending_tool_hint["_buf"] += delta
                match = _ARG_SNIFF_RE.search(_pending_tool_hint["_buf"])
                if match:
                    snippet = match.group(1)
                    if len(snippet) > 80:
                        snippet = snippet[:80]
                    print(f"  ({snippet})", end="", flush=True)
                    _pending_tool_hint["printed_arg"] = True

    elif event_type == "tool_call":
        _flush_tool_hint()
        if _thinking_open:
            print()
            _thinking_open = False
        if event["name"] in _CONTROL_TOOLS:
            return
        safe_args = _safe_tool_args_for_display(event["name"], event["args"])
        _print_block("工具", f"{event['name']}: {safe_args}")

    elif event_type == "tool_result":
        _flush_tool_hint()
        if _thinking_open:
            print()
            _thinking_open = False
        result = event.get("result", "")
        if result.startswith("CHAT:"):
            return
        if result.startswith("DECIDE:"):
            return
        preview = _format_tool_result_preview(result)
        _print_block("工具结果", preview)

    elif event_type == "chat_request":
        _flush_tool_hint()
        return

    elif event_type == "node_transition":
        _flush_tool_hint()
        if _thinking_open:
            print()
            _thinking_open = False
        from_display = event.get("from_display", event.get("from", "未知"))
        to_display = event.get("to_display", event.get("to", "未知"))
        route_text = f"{from_display} -> {to_display}"
        content_preview = event.get("content_preview", "").strip()
        if content_preview:
            route_text += f"\n[指令] {content_preview}"
        _print_block("路由", route_text)

    elif event_type == "error":
        _flush_tool_hint()
        if _thinking_open:
            print()
            _thinking_open = False
        _print_block("错误", event["message"])

    elif event_type == "completed":
        _flush_tool_hint()
        if _thinking_open:
            print()
            _thinking_open = False
        _print_block("完成", "已完成")


def _flush_tool_hint() -> None:
    """End the pending tool-hint line if one is open."""
    global _pending_tool_hint
    if _pending_tool_hint is not None:
        print(flush=True)
        _pending_tool_hint = None


def _node_display_name(node: dict | None, fallback: str | None = None) -> str:
    if not node:
        return fallback or "未知"
    return str(node.get("display_name") or node.get("name") or fallback or "未知")


def _route_preview_text(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return ""
    text = text.replace("\r", " ").replace("\n", " | ")
    if len(text) > _ROUTE_PREVIEW_CHARS:
        return text[:_ROUTE_PREVIEW_CHARS]
    return text


def _format_tool_result_preview(result: str) -> str:
    text = result if isinstance(result, str) else str(result)
    return build_range_preview(
        text,
        _TOOL_RESULT_PREVIEW_CHARS,
        **extract_preview_ranges(text),
    )


def _print_block(label: str, body: str) -> None:
    section_bar = _section_bar()
    print()
    print(f"[{label}] {body}")
    print(section_bar)


def _safe_tool_args_for_display(tool_name: str, args: Any) -> Any:
    if not isinstance(args, dict):
        return args
    safe = dict(args)
    if tool_name in {"write_file", "append_file"} and "content" in safe:
        content = safe.get("content", "")
        content_len = len(content) if isinstance(content, str) else 0
        safe["content"] = f"<内容 {content_len} 字符>"
    if tool_name == "apply_diff":
        for key in ("search", "replace"):
            if key in safe and isinstance(safe[key], str):
                text = safe[key]
                safe[key] = f"<{key} {len(text)} 字符>"
    if tool_name == "apply_patch" and "patch" in safe and isinstance(safe["patch"], str):
        safe["patch"] = f"<补丁 {len(safe['patch'])} 字符>"
    return safe


def _section_bar() -> str:
    width = shutil.get_terminal_size(fallback=(120, 30)).columns
    bar_len = max(40, width - 2)
    return "-" * bar_len


__all__ = ["run"]
