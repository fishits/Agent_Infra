"""Local vLLM Responses API adapter.

This adapter calls an OpenAI-compatible /v1/responses endpoint directly and
normalizes vLLM/Qwen tool-call output into the Responses event shape consumed by
the framework runtime.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import requests


class _EventItem(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def _load_env() -> None:
    env_files = [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",
    ]
    for env_file in env_files:
        if not env_file.exists():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())
        return


_load_env()


def local_responses_llm(
    model: str = "qwen3.6",
    reasoning_effort: str = "",
    reasoning_summary: str = "",
    tool_choice: str = "required",
):
    del reasoning_effort, reasoning_summary

    base_url = (os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:5000/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY") or "EMPTY"
    timeout = float(os.getenv("LOCAL_LLM_TIMEOUT", "300"))
    max_output_tokens = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "6000"))

    def llm(
        input_items: list[dict],
        tools: list[dict] | None = None,
        instructions: str = "",
    ) -> Iterator[Any]:
        payload: dict[str, Any] = {
            "model": model,
            "input": input_items,
            "max_output_tokens": max_output_tokens,
            "tool_choice": tool_choice,
        }
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = _normalize_responses_tools(tools)
            payload["instructions"] = _with_xml_tool_protocol(
                payload.get("instructions", ""),
                payload["tools"],
            )

        enable_thinking = os.getenv("LOCAL_LLM_ENABLE_THINKING", "true").lower()
        if enable_thinking in {"0", "false", "no", "off"}:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        started_at = time.perf_counter()
        response = requests.post(
            f"{base_url}/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        _print_metrics(data, started_at, base_url)
        yield from _emit_events(data, payload.get("tools") or [])

    return llm


def _normalize_responses_tools(tools: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        fn = tool.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if not name:
                continue
            normalized.append(
                {
                    "type": "function",
                    "name": name,
                    "description": fn.get("description") or tool.get("description", ""),
                    "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                }
            )
            continue
        name = tool.get("name")
        if name:
            normalized.append(
                {
                    "type": "function",
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                }
            )
    return normalized


def _with_xml_tool_protocol(instructions: str, tools: list[dict]) -> str:
    protocol = (
        "You have function tools available. When you call a tool, do not answer in prose. "
        "Output exactly one XML tool call using this format:\n"
        "<tool_call>\n"
        "<function=tool_name>\n"
        "<parameter=arg_name>arg_value</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "Available tools:\n"
        f"{json.dumps(tools, ensure_ascii=False)}"
    )
    if instructions:
        return f"{instructions}\n\n{protocol}"
    return protocol


def _emit_events(data: dict[str, Any], tools: list[dict]) -> Iterator[Any]:
    response_id = data.get("id") or f"resp_{uuid.uuid4().hex}"
    output_items: list[Any] = []
    tool_names = {tool.get("name") for tool in tools if isinstance(tool, dict)}

    for item in data.get("output") or []:
        item_type = item.get("type")
        if item_type == "reasoning":
            text = _extract_text(item.get("content"))
            if text:
                yield SimpleNamespace(type="response.reasoning_summary_text.delta", delta=text)
                for parsed in _parse_qwen_tool_calls(text, tool_names):
                    output_items.append(parsed)
                    yield from _yield_function_events(parsed)
            continue

        if item_type == "function_call":
            event_item = _function_item_from_response(item)
            output_items.append(event_item)
            yield from _yield_function_events(event_item)
            continue

        text = _extract_text(item.get("content"))
        for parsed in _parse_qwen_tool_calls(text, tool_names):
            output_items.append(parsed)
            yield from _yield_function_events(parsed)

    yield SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            id=response_id,
            output=output_items,
            usage=_to_usage(data.get("usage")),
        ),
    )


def _yield_function_events(item: _EventItem) -> Iterator[Any]:
    yield SimpleNamespace(type="response.output_item.added", item=item)
    yield SimpleNamespace(
        type="response.function_call_arguments.delta",
        item_id=item.id,
        delta=item.arguments,
    )
    yield SimpleNamespace(type="response.output_item.done", item=item)


def _function_item_from_response(item: dict[str, Any]) -> _EventItem:
    item_id = item.get("id") or f"fc_{uuid.uuid4().hex}"
    call_id = item.get("call_id") or f"call_{uuid.uuid4().hex}"
    return _EventItem(
        type="function_call",
        id=item_id,
        call_id=call_id,
        name=item.get("name") or "unknown_tool",
        arguments=item.get("arguments") or "{}",
        status=item.get("status") or "completed",
    )


_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_FUNCTION_RE = re.compile(r"<function=([A-Za-z_][\w.-]*)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([A-Za-z_][\w.-]*)>(.*?)</parameter>", re.DOTALL)


def _parse_qwen_tool_calls(text: str, tool_names: set[str | None]) -> list[_EventItem]:
    parsed: list[_EventItem] = []
    if not text:
        return parsed

    for tool_block in _TOOL_CALL_RE.findall(text):
        fn_match = _FUNCTION_RE.search(tool_block)
        if not fn_match:
            continue
        name = fn_match.group(1).strip()
        if tool_names and name not in tool_names:
            continue
        args: dict[str, str] = {}
        for param, value in _PARAM_RE.findall(fn_match.group(2)):
            args[param.strip()] = value.strip()
        item = _EventItem(
            type="function_call",
            id=f"fc_{uuid.uuid4().hex}",
            call_id=f"call_{uuid.uuid4().hex}",
            name=name,
            arguments=json.dumps(args, ensure_ascii=False),
            status="completed",
        )
        parsed.append(item)
    return parsed


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            text = block.get("text")
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def _to_usage(usage: dict[str, Any] | None):
    if not isinstance(usage, dict):
        return None
    return SimpleNamespace(
        input_tokens=usage.get("input_tokens") or 0,
        output_tokens=usage.get("output_tokens") or 0,
        total_tokens=usage.get("total_tokens") or 0,
    )


def _print_metrics(data: dict[str, Any], started_at: float, base_url: str) -> None:
    elapsed = time.perf_counter() - started_at
    usage = data.get("usage") if isinstance(data, dict) else {}
    input_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    output_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    cached_tokens = None
    if isinstance(usage, dict) and isinstance(usage.get("input_tokens_details"), dict):
        value = usage["input_tokens_details"].get("cached_tokens")
        cached_tokens = value if isinstance(value, int) else None
    decode_rate = f"{output_tokens / elapsed:.2f}" if isinstance(output_tokens, int) and elapsed else "n/a"
    del base_url
    print(
        "\n[LLM] "
        "TTFT=n/a "
        f"time={elapsed:.3f}s "
        f"in={input_tokens if input_tokens is not None else 'n/a'} "
        f"out={output_tokens if output_tokens is not None else 'n/a'} "
        f"cache={cached_tokens if cached_tokens is not None else 'n/a'} "
        "prefill=n/a "
        f"decode={decode_rate} tok/s "
        f"status={data.get('status')}",
        flush=True,
    )


__all__ = ["local_responses_llm"]
