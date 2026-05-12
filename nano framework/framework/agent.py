"""Agent runtime.

This file runs one agent loop and the node adapter around it.
Context persistence is delegated to mem.py.
Context assembly is delegated to context.py.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator
from datetime import datetime
import os

from .big_todo import pre_tool_gate
from .context import build_openai_input_items
from .control_tools import render_decide_rules
from .mem import append_items
from .tools import call_tool, convert_to_openai_tools, parse_args


_NO_TOOL_CALL_WARNING = (
    "助手的每一次回复都必须调用工具.任务完成时请使用 new_decide(...)进行路由;任务未完成请继续使用工具完成任务;需要和用户沟通时使用chat(...)"
)
_MAX_NO_TOOL_RETRIES = 3
_MAX_LENGTH_CONTINUATIONS = 3
_LENGTH_CONTINUE_PROMPT = (
    "上一条回复因为输出达到上限而中断。不要重复前文，直接从中断处继续，"
    "并优先尽快完成本轮需要的工具调用。"
)
_CONTROL_TOOL_NAMES = {"chat", "decide", "new_decide", "end_decide"}


def agent_loop(node: dict, input_items: list[dict]) -> Iterator[dict]:
    """Run one generic agent loop from one node definition."""
    current_items = list(input_items)
    attempt = 1
    length_continuations = 0

    while attempt <= _MAX_NO_TOOL_RETRIES:
        reasoning_text = ""
        response_stream = node["llm"](
            input_items=current_items,
            tools=convert_to_openai_tools(node["tools"]),
            instructions=_build_instructions(node),
        )
        response_output = []
        tool_calls: dict[str, dict] = {}
        response_id: str | None = None
        finish_reason: str | None = None

        for event in response_stream:
            event_type = event.type

            if event_type == "response.reasoning_summary_text.delta":
                reasoning_text += str(event.delta or "")
                yield {"type": "reasoning", "content": event.delta}

            elif event_type == "response.output_item.added" and event.item.type == "function_call":
                tool_calls[event.item.id] = {
                    "name": event.item.name,
                    "call_id": event.item.call_id,
                    "item_id": event.item.id,
                    "arguments": "",
                }
                yield {"type": "tool_call_start", "name": event.item.name}

            elif event_type == "response.function_call_arguments.delta":
                tool_call = tool_calls.get(event.item_id)
                if tool_call is None:
                    continue
                tool_call["arguments"] += event.delta
                yield {"type": "tool_call_args", "name": tool_call["name"], "delta": event.delta}

            elif event_type == "response.output_item.done" and hasattr(event, "item"):
                if getattr(event.item, "type", None) == "function_call" and reasoning_text:
                    setattr(event.item, "thinking", reasoning_text)
                response_output.append(event.item)
                yield {"type": "response_item", "item": event.item}

            elif event_type in {"response.completed", "response.done"} and hasattr(event, "response"):
                response_id = getattr(event.response, "id", None)
                finish_reason = getattr(event.response, "finish_reason", None)
                usage = getattr(event.response, "usage", None)
                usage_record = _build_usage_record(usage, response_id)
                if usage_record is not None:
                    yield {"type": "response_usage", "usage": usage_record}
                if not response_output and event.response.output:
                    response_output = list(event.response.output)
                    for item in response_output:
                        yield {"type": "response_item", "item": item}

        if not tool_calls:
            if finish_reason == "length" and length_continuations < _MAX_LENGTH_CONTINUATIONS:
                length_continuations += 1
                yield {
                    "type": "warning",
                    "attempt": attempt,
                    "message": "model output hit max_tokens; auto-continuing from the interruption point",
                }
                current_items = current_items + _response_items_to_input_items(response_output) + [
                    {"role": "user", "content": _LENGTH_CONTINUE_PROMPT}
                ]
                continue
            yield {"type": "warning", "attempt": attempt, "message": _NO_TOOL_CALL_WARNING}
            if attempt < _MAX_NO_TOOL_RETRIES:
                current_items = current_items + [{"role": "user", "content": _NO_TOOL_CALL_WARNING}]
                attempt += 1
                continue
            yield {
                "type": "error",
                "message": f"模型连续 {_MAX_NO_TOOL_RETRIES} 轮都没有调用工具，当前节点执行已停止。",
            }
            yield {"type": "next_node", "node": None}
            return

        break

    next_node = None
    next_content = ""
    clear_target_memory = False
    tool_outputs: list[dict] = []

    parsed_tool_calls = [
        {**tool_call, "args": parse_args(tool_call["arguments"])}
        for tool_call in tool_calls.values()
    ]
    for tool_call in parsed_tool_calls:
        yield {"type": "tool_call", "name": tool_call["name"], "args": tool_call["args"]}

    executed_results = _execute_tool_calls(node, parsed_tool_calls)

    for tool_call, result in zip(parsed_tool_calls, executed_results):
        yield {"type": "tool_result", "name": tool_call["name"], "result": result}

        context_result = _tool_result_for_context(tool_call["name"], result)
        output_item = {
            "type": "function_call_output",
            "call_id": tool_call["call_id"],
            "output": context_result,
        }
        tool_outputs.append(output_item)
        yield {"type": "tool_output_item", "item": output_item}

        control = _parse_control_result(result)
        if control is None:
            continue

        if control["type"] == "chat":
            yield {
                "type": "chat_request",
                "message": control["message"],
                "context_items": response_output + tool_outputs,
                "response_id": response_id,
            }
            return

        next_node = control["target"]
        next_content = _forward_content(
            run_id=node["run_id"],
            node_name=node["name"],
            target_node=next_node,
            content=control["content"],
        )
        clear_target_memory = bool(control.get("clear_target_memory"))
        break

    yield {
        "type": "next_node",
        "node": (next_node, next_content) if next_node else None,
        "clear_target_memory": clear_target_memory if next_node else False,
        "context_items": response_output + tool_outputs,
        "response_id": response_id,
    }


def node_invoke(graph: dict, node: dict, message: str = "") -> Iterator[dict]:
    """Run one node turn from graph state and one new message."""
    run_id = graph["state"]["run_id"]
    node_name = node["name"]

    yield {
        "type": "node_start",
        "node": node_name,
        "display_name": node.get("display_name") or node_name,
        "message": (message or "").strip(),
        "root_message": (graph["state"].get("root_user_message") or "").strip(),
    }
    context = build_openai_input_items(run_id, node_name)

    new_items: list = []
    if message:
        new_items = [{"role": "user", "content": message}]
        append_items(run_id, node_name, new_items)
        yield {"type": "user_message", "content": message}

    events = agent_loop(
        node={**node, "run_id": run_id, "graph_state": graph["state"]},
        input_items=context + new_items,
    )

    for event in events:
        event_type = event["type"]

        if event_type == "response_item":
            append_items(run_id, node_name, [event["item"]])
            yield event
            continue

        if event_type == "response_usage":
            append_items(run_id, node_name, [event["usage"]])
            yield event
            continue

        if event_type == "tool_output_item":
            append_items(run_id, node_name, [event["item"]])
            yield event
            continue

        if event_type in ("chat_request", "next_node"):
            yield event
            return

        yield event


def invoke_stream(node: dict, run_id: str, user_message: str = "") -> Iterator[dict]:
    """Compatibility wrapper around node_invoke()."""
    graph = {"nodes": [node], "state": {"run_id": run_id, "current_node": node["name"]}}
    yield from node_invoke(graph, node, user_message)


def _parse_control_result(result: str) -> dict | None:
    """Interpret control-flow tool outputs."""
    if result.startswith("DECIDE:"):
        target, _, content = result[7:].partition(":")
        return {"type": "decide", "target": target, "content": content, "clear_target_memory": False}
    if result.startswith("NEW_DECIDE:"):
        target, _, content = result[11:].partition(":")
        return {"type": "decide", "target": target, "content": content, "clear_target_memory": True}
    if result.startswith("CHAT:"):
        return {"type": "chat", "message": result[5:]}
    return None


def _tool_result_for_context(tool_name: str, result: str) -> str:
    return result


def _execute_tool_calls(node: dict, tool_calls: list[dict]) -> list[str]:
    """Execute same-response tool calls concurrently when it is safe to do so."""
    if len(tool_calls) <= 1 or any(call["name"] in _CONTROL_TOOL_NAMES for call in tool_calls):
        return [_execute_one_tool_call(node, tool_call) for tool_call in tool_calls]

    results: list[str | None] = [None] * len(tool_calls)
    with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
        futures = {
            executor.submit(_execute_one_tool_call, node, tool_call): index
            for index, tool_call in enumerate(tool_calls)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [result if result is not None else "tool execution failed" for result in results]


def _execute_one_tool_call(node: dict, tool_call: dict) -> str:
    runtime_context = node.get("graph_state", {})
    if isinstance(runtime_context, dict):
        runtime_context.setdefault("run_id", node["run_id"])
        runtime_context["node_name"] = node["name"]
        runtime_context.setdefault("cwd", os.getcwd())
    tool_name = tool_call["name"]
    if _big_todo_enabled(node["tools"]) and tool_name not in _CONTROL_TOOL_NAMES:
        gate = pre_tool_gate(tool_name, runtime_context.get("run_id"), runtime_context.get("node_name"))
        if gate is not None:
            return gate
    return call_tool(
        node["tools"],
        tool_name,
        tool_call["args"],
        runtime_context=runtime_context,
    )


def _big_todo_enabled(tools: list) -> bool:
    tool_names = {fn.__name__ for fn in tools}
    return "todo" in tool_names and "end_decide" in tool_names


def _build_instructions(node: dict) -> str:
    """Build node instructions from its static definition."""
    prompt = node["prompt"]
    if callable(prompt):
        prompt = prompt(node.get("graph_state", {}))
    edges = node.get("edges")
    if isinstance(edges, dict):
        return prompt + "\n" + render_decide_rules(edges)
    return prompt


def _forward_content(run_id: str, node_name: str, target_node: str, content: str) -> str:
    """Tag node-to-node content with its source node."""
    del run_id, target_node
    body = (content or "").strip()
    if not body:
        return f"[{node_name}]"
    return f"[{node_name}]\n{body}"


def _build_usage_record(usage: object, response_id: str | None) -> dict | None:
    """Build one persistent token-usage record from a completed response."""
    if usage is None:
        return None

    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if not all(isinstance(value, int) for value in (input_tokens, output_tokens, total_tokens)):
        return None

    record = {
        "type": "token_usage",
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "response_id": response_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    ttft = getattr(usage, "ttft", None)
    total_time = getattr(usage, "total_time", None)
    decode_tokens_per_second = getattr(usage, "decode_tokens_per_second", None)
    cached_tokens = getattr(usage, "cached_tokens", None)
    cache_text = getattr(usage, "cache_text", None)
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


def _response_items_to_input_items(items: list[object]) -> list[dict]:
    projected: list[dict] = []
    for item in items:
        raw = None
        if isinstance(item, dict):
            raw = dict(item)
        elif hasattr(item, "to_dict"):
            raw = item.to_dict()
        elif hasattr(item, "__dict__"):
            raw = dict(item.__dict__)
        if not isinstance(raw, dict):
            continue
        role = raw.get("role")
        item_type = raw.get("type")
        if role in {"user", "assistant"} and raw.get("content"):
            projected.append({"role": role, "content": raw.get("content"), **({"thinking": raw["thinking"]} if raw.get("thinking") else {})})
        elif item_type == "function_call":
            projected.append(
                {
                    "type": "function_call",
                    "call_id": raw.get("call_id"),
                    "name": raw.get("name"),
                    "arguments": raw.get("arguments"),
                    **({"thinking": raw["thinking"]} if raw.get("thinking") else {}),
                }
            )
        elif item_type == "function_call_output":
            projected.append(
                {
                    "type": "function_call_output",
                    "call_id": raw.get("call_id"),
                    "output": raw.get("output"),
                }
            )
    return projected


__all__ = ["agent_loop", "invoke_stream", "node_invoke"]
