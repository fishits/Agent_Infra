"""Isolated Chat Completions adapter for local SGLang servers.

This file is intentionally separate from framework.llm_local. It translates the
framework's Responses-style memory into Chat Completions messages before sending
requests to SGLang.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from openai import APITimeoutError, InternalServerError, OpenAI
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


def local_chat_llm(
    model: str = "qwen3-coder-next",
    reasoning_effort: str = "",
    reasoning_summary: str = "",
    tool_choice: str = "required",
):
    del reasoning_effort, reasoning_summary

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY") or "EMPTY",
        base_url=os.getenv("OPENAI_BASE_URL") or "http://127.0.0.1:8000/v1",
        timeout=float(os.getenv("LOCAL_LLM_TIMEOUT", "120")),
        max_retries=0,
    )

    def llm(
        input_items: list[dict],
        tools: list[dict] | None = None,
        instructions: str = "",
    ):
        messages = _to_chat_messages(input_items=input_items, instructions=instructions)
        normalized_tools = _to_chat_tools(tools)
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "6000")),
        }
        enable_thinking = os.getenv("LOCAL_LLM_ENABLE_THINKING", "false").lower()
        if enable_thinking in {"0", "false", "no", "off"}:
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        if normalized_tools:
            kwargs["tools"] = normalized_tools
            kwargs["tool_choice"] = tool_choice

        debug_info = {
            "model": model,
            "messages_count": len(messages),
            "tools_count": len(normalized_tools),
            "tool_choice": kwargs.get("tool_choice"),
            "base_url": str(client.base_url),
        }
        for attempt in range(1, 6):
            try:
                t0 = time.perf_counter()
                completion = _collect_streaming_completion(
                    stream=client.chat.completions.create(**kwargs),
                    model=model,
                    started_at=t0,
                    debug_info=debug_info,
                    cache_before=None,
                )
                return _emit_responses_style_events(completion)
            except InternalServerError as exc:
                if attempt == 5:
                    print(f"\n[Local Chat LLM Error] final failure, request={debug_info}")
                    raise
                wait = 2**attempt
                print(f"\n[Local Chat LLM] server error attempt {attempt}/5: {exc}; retry in {wait}s")
                time.sleep(wait)
            except APITimeoutError:
                print(f"\n[Local Chat LLM Timeout] request exceeded timeout, request={debug_info}")
                raise
            except Exception:
                print(f"\n[Local Chat LLM Error] request={debug_info}")
                raise

    return llm


def _collect_streaming_completion(
    stream,
    model: str,
    started_at: float,
    debug_info: dict,
    cache_before: tuple[int, int] | None,
    cache_before_ms: float = 0.0,
):
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    usage = None
    response_id = f"resp_{uuid.uuid4().hex}"
    first_token_at: float | None = None

    for chunk in stream:
        if getattr(chunk, "id", None):
            response_id = chunk.id

        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        delta = choices[0].delta
        delta_reasoning = _delta_reasoning_text(delta)
        if delta_reasoning:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            reasoning_parts.append(delta_reasoning)

        delta_content = getattr(delta, "content", None)
        if delta_content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            content_parts.append(delta_content)

        for tool_delta in getattr(delta, "tool_calls", None) or []:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            index = getattr(tool_delta, "index", 0) or 0
            existing = tool_calls.setdefault(
                index,
                {
                    "id": getattr(tool_delta, "id", None) or f"call_{uuid.uuid4().hex}",
                    "name": "",
                    "arguments": "",
                },
            )
            if getattr(tool_delta, "id", None):
                existing["id"] = tool_delta.id
            func = getattr(tool_delta, "function", None)
            if func is not None:
                if getattr(func, "name", None):
                    existing["name"] = func.name
                if getattr(func, "arguments", None):
                    existing["arguments"] += func.arguments

    ended_at = time.perf_counter()
    cache_after_started = time.perf_counter()
    cache_after = _read_vllm_cache_metrics(debug_info["base_url"])
    cache_after_ms = (time.perf_counter() - cache_after_started) * 1000
    ttft = (first_token_at or ended_at) - started_at
    total_time = ended_at - started_at
    _print_metrics(
        usage=usage,
        ttft=ttft,
        total_time=total_time,
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        tool_calls=list(tool_calls.values()),
        debug_info=debug_info,
        cache_delta=_cache_delta(cache_before, cache_after),
        cache_snapshot=cache_after,
        cache_metrics_ms=cache_before_ms + cache_after_ms,
    )

    message = SimpleNamespace(
        role="assistant",
        content="".join(content_parts) or None,
        reasoning_content="".join(reasoning_parts) or None,
        tool_calls=[
            SimpleNamespace(
                id=data["id"],
                type="function",
                function=SimpleNamespace(
                    name=data["name"] or "unknown_tool",
                    arguments=data["arguments"] or "{}",
                ),
            )
            for _, data in sorted(tool_calls.items())
        ] or None,
    )
    return SimpleNamespace(
        id=response_id,
        model=model,
        choices=[SimpleNamespace(index=0, message=message)],
        usage=usage,
    )


def _print_metrics(
    usage,
    ttft: float,
    total_time: float,
    content: str,
    reasoning: str,
    tool_calls: list[dict],
    debug_info: dict,
    cache_delta: tuple[int, int] | None,
    cache_snapshot: tuple[int, int] | None,
    cache_metrics_ms: float = 0.0,
) -> None:
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    cached_tokens = _cached_tokens_from_chat_usage(usage)
    if cached_tokens is None and cache_delta is not None:
        cached_tokens = cache_delta[0]
    cache_text = _format_cache(cache_delta, cache_snapshot, cached_tokens, prompt_tokens)
    decode_time = max(total_time - ttft, 1e-6)
    decode_tok_s = (
        f"{completion_tokens / decode_time:.2f}" if isinstance(completion_tokens, int) else "n/a"
    )
    visible_chars = len(content or "")
    reasoning_chars = len(reasoning or "")
    tool_count = len(tool_calls or [])
    print(
        "\n[LLM] "
        f"TTFT={ttft:.3f}s "
        f"time={total_time:.3f}s "
        f"decode={decode_tok_s} tok/s "
        f"in={prompt_tokens if prompt_tokens is not None else 'n/a'} "
        f"out={completion_tokens if completion_tokens is not None else 'n/a'} "
        f"cache={cache_text}",
        flush=True,
    )
    if (
        isinstance(completion_tokens, int)
        and completion_tokens > 0
        and not visible_chars
        and not reasoning_chars
        and not tool_count
    ):
        print(
            "[Local Chat LLM Warning] completion tokens were produced, but no visible content, reasoning_content, or tool_calls were streamed. "
            "This usually means the local server emitted hidden thinking in a field this adapter cannot see, or the model exhausted max_tokens before producing a tool call.",
            flush=True,
        )


_VLLM_CACHE_RE = re.compile(
    r'^vllm:(prefix_cache_queries|prefix_cache_hits)_total\{[^}]*model_name="[^"]+"[^}]*\}\s+([0-9.eE+-]+)$',
    re.MULTILINE,
)


def _read_vllm_cache_metrics(base_url: str) -> tuple[int, int] | None:
    if not _env_enabled("LOCAL_LLM_ENABLE_METRICS", default=False):
        return None
    metrics_url = base_url.rstrip("/")
    if metrics_url.endswith("/v1"):
        metrics_url = metrics_url[:-3]
    metrics_url = f"{metrics_url}/metrics"
    try:
        response = requests.get(
            metrics_url,
            timeout=float(os.getenv("LOCAL_LLM_METRICS_TIMEOUT", "0.25")),
        )
        response.raise_for_status()
    except Exception:
        return None

    values: dict[str, int] = {}
    for name, raw_value in _VLLM_CACHE_RE.findall(response.text):
        values[name] = int(float(raw_value))
    queries = values.get("prefix_cache_queries")
    hits = values.get("prefix_cache_hits")
    if queries is None or hits is None:
        return None
    return hits, queries


def _cache_delta(
    before: tuple[int, int] | None,
    after: tuple[int, int] | None,
) -> tuple[int, int] | None:
    if before is None or after is None:
        return None
    hits = max(after[0] - before[0], 0)
    queries = max(after[1] - before[1], 0)
    return hits, queries


def _format_cache(
    cache_delta: tuple[int, int] | None,
    cache_snapshot: tuple[int, int] | None,
    cached_tokens: int | None,
    prompt_tokens: int | None,
) -> str:
    if cached_tokens is not None:
        if isinstance(prompt_tokens, int) and prompt_tokens > 0:
            return f"{cached_tokens}/{prompt_tokens}={cached_tokens / prompt_tokens:.1%}"
        return str(cached_tokens)
    if cache_delta is not None:
        hits, queries = cache_delta
        if queries > 0:
            return f"{hits}/{queries}={hits / queries:.1%}"
        return f"{hits}/0"
    if cache_snapshot is not None:
        hits, queries = cache_snapshot
        if queries > 0:
            return f"{hits}/{queries}={hits / queries:.1%}"
        return f"{hits}/0"
    return "n/a"


def _cached_tokens_from_chat_usage(usage) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
    if details is None:
        return None
    if isinstance(details, dict):
        value = details.get("cached_tokens")
    else:
        value = getattr(details, "cached_tokens", None)
    return value if isinstance(value, int) else None


def _env_enabled(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off", ""}


def _to_chat_messages(input_items: list[dict], instructions: str) -> list[dict]:
    messages: list[dict] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    for item in input_items:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        item_type = item.get("type")
        if role in {"system", "user", "assistant"}:
            messages.append({"role": role, "content": _content_to_text(item.get("content", ""))})
            continue

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": item.get("name", "unknown_tool"),
                                "arguments": item.get("arguments") or "{}",
                            },
                        }
                    ],
                }
            )
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if call_id:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": str(item.get("output", "")),
                    }
                )

    return messages


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")

    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in {"input_text", "output_text", "text"}:
            parts.append(str(block.get("text", "")))
        elif hasattr(block, "text"):
            parts.append(str(block.text))
    return "\n".join(part for part in parts if part)


def _delta_reasoning_text(delta) -> str:
    for attr in ("reasoning_content", "reasoning", "reasoning_text"):
        value = getattr(delta, attr, None)
        if isinstance(value, str) and value:
            return value
    return ""


def _to_chat_tools(tools: list[dict] | None) -> list[dict]:
    if not tools:
        return []

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
                    "function": {
                        "name": name,
                        "description": fn.get("description") or tool.get("description", ""),
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )
            continue

        name = tool.get("name")
        if name:
            normalized.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
                    },
                }
            )

    return normalized


def _emit_responses_style_events(completion):
    choice = completion.choices[0]
    message = choice.message
    output_items = []

    reasoning_text = getattr(message, "reasoning_content", None)
    if reasoning_text:
        yield SimpleNamespace(type="response.reasoning_summary_text.delta", delta=reasoning_text)

    tool_calls = getattr(message, "tool_calls", None) or []
    for tool_call in tool_calls:
        func = tool_call.function
        item_id = f"fc_{uuid.uuid4().hex}"
        call_id = getattr(tool_call, "id", None) or f"call_{uuid.uuid4().hex}"
        item = _EventItem(
            type="function_call",
            id=item_id,
            call_id=call_id,
            name=getattr(func, "name", "unknown_tool"),
            arguments=getattr(func, "arguments", "{}") or "{}",
            status="completed",
        )
        output_items.append(item)
        yield SimpleNamespace(type="response.output_item.added", item=item)
        yield SimpleNamespace(type="response.function_call_arguments.delta", item_id=item_id, delta=item.arguments)
        yield SimpleNamespace(type="response.output_item.done", item=item)

    if not tool_calls:
        text = _content_to_text(getattr(message, "content", "") or "")
        if text:
            item = _EventItem(
                type="message",
                role="assistant",
                content=text,
            )
            output_items.append(item)
            yield SimpleNamespace(type="response.output_item.done", item=item)
            preview = text[:300].replace("\r", "\\r").replace("\n", "\\n")
            print(
                "\n[Local Chat LLM Warning] "
                f"no tool calls returned; captured assistant content chars={len(text)} "
                f"preview={preview}",
                flush=True,
            )

    yield SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            id=getattr(completion, "id", f"resp_{uuid.uuid4().hex}"),
            output=output_items,
            usage=_to_responses_usage(getattr(completion, "usage", None)),
            finish_reason=getattr(choice, "finish_reason", None),
        ),
    )


def _to_responses_usage(usage):
    if usage is None:
        return None
    return SimpleNamespace(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
    )


__all__ = ["local_chat_llm"]
