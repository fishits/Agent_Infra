"""DeepSeek Chat Completions adapter.

DeepSeek V4 uses Chat Completions as the primary API surface. In thinking mode,
assistant messages that include tool calls must be replayed back with their
original ``reasoning_content`` on the follow-up request. The generic
``llm_local_chat`` adapter intentionally flattens chat responses into a
Responses-style event stream, which drops that replay-critical assistant
message body.

This adapter consumes the framework context projection and sends DeepSeek-native
Chat Completions requests. The context layer is responsible for storing the
assistant tool-call reasoning under a ``thinking`` field; this adapter maps that
field back to DeepSeek's ``reasoning_content``.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI


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


def deepseek_llm(
    model: str = "deepseek-v4-flash",
    reasoning_effort: str = "",
    reasoning_summary: str = "",
    tool_choice: str = "auto",
):
    del reasoning_summary

    client = OpenAI(
        api_key=(
            os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("ONLYTEAM_API_KEY")
        ),
        base_url=(
            os.getenv("DEEPSEEK_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("ONLYTEAM_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/"),
        timeout=float(os.getenv("LOCAL_LLM_TIMEOUT", "120")),
        max_retries=0,
    )

    def llm(
        input_items: list[dict],
        tools: list[dict] | None = None,
        instructions: str = "",
    ):
        normalized_tools = _to_chat_tools(tools)
        projected_messages = _project_input_items(input_items, instructions)

        kwargs = {
            "model": model,
            "messages": projected_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "max_tokens": int(os.getenv("LOCAL_LLM_MAX_TOKENS", "6000")),
            "reasoning_effort": _normalize_reasoning_effort(reasoning_effort),
            "extra_body": {"thinking": {"type": _thinking_mode()}},
        }
        if normalized_tools:
            kwargs["tools"] = normalized_tools
            kwargs["tool_choice"] = tool_choice

        debug_info = {
            "model": model,
            "messages_count": len(projected_messages),
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
                )
                return _emit_responses_style_events(completion)
            except InternalServerError as exc:
                if attempt == 5:
                    print(f"\n[DeepSeek LLM Error] final failure, request={debug_info}")
                    raise
                wait = 2**attempt
                print(f"\n[DeepSeek LLM] server error attempt {attempt}/5: {exc}; retry in {wait}s")
                time.sleep(wait)
            except APIConnectionError as exc:
                if attempt == 5:
                    print(f"\n[DeepSeek LLM Error] final connection failure, request={debug_info}")
                    raise
                wait = 2**attempt
                print(f"\n[DeepSeek LLM] connection error attempt {attempt}/5: {exc}; retry in {wait}s")
                time.sleep(wait)
            except APITimeoutError:
                print(f"\n[DeepSeek LLM Timeout] request exceeded timeout, request={debug_info}")
                raise
            except Exception:
                print(f"\n[DeepSeek LLM Error] request={debug_info}")
                raise

    return llm


def _project_input_items(input_items: list[dict], instructions: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if instructions:
        messages.append({"role": "system", "content": instructions})

    index = 0
    while index < len(input_items):
        item = input_items[index]
        if not isinstance(item, dict):
            index += 1
            continue

        role = item.get("role")
        item_type = item.get("type")
        if role in {"system", "user", "assistant"}:
            projected = {"role": role, "content": _content_to_text(item.get("content", ""))}
            thinking = item.get("thinking")
            if role == "assistant" and isinstance(thinking, str) and thinking:
                projected["reasoning_content"] = thinking
            messages.append(projected)
            index += 1
            continue

        if item_type == "function_call":
            tool_calls: list[dict[str, Any]] = []
            thinking_parts: list[str] = []
            while index < len(input_items):
                grouped = input_items[index]
                if not isinstance(grouped, dict) or grouped.get("type") != "function_call":
                    break
                call_id = grouped.get("call_id") or grouped.get("id") or f"call_{uuid.uuid4().hex}"
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": grouped.get("name", "unknown_tool"),
                            "arguments": grouped.get("arguments") or "{}",
                        },
                    }
                )
                thinking = grouped.get("thinking")
                if isinstance(thinking, str) and thinking and thinking not in thinking_parts:
                    thinking_parts.append(thinking)
                index += 1
            projected = {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls,
            }
            if thinking_parts:
                projected["reasoning_content"] = "\n\n".join(thinking_parts)
            messages.append(projected)
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
            index += 1
            continue

        index += 1

    return messages


def _collect_streaming_completion(
    stream,
    model: str,
    started_at: float,
    debug_info: dict,
):
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    usage = None
    response_id = f"resp_{uuid.uuid4().hex}"
    first_token_at: float | None = None
    finish_reason: str | None = None

    for chunk in stream:
        if getattr(chunk, "id", None):
            response_id = chunk.id

        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = chunk_usage

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            continue

        choice = choices[0]
        if getattr(choice, "finish_reason", None):
            finish_reason = choice.finish_reason
        delta = choice.delta
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
    ttft = (first_token_at or ended_at) - started_at
    total_time = ended_at - started_at
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    decode_tokens_per_second = (
        completion_tokens / max(total_time - ttft, 1e-6)
        if isinstance(completion_tokens, int) and completion_tokens > 0 and total_time > ttft
        else 0.0
    )
    cached_tokens = _cached_tokens_from_chat_usage(usage)
    cache_text = _format_cache(cached_tokens, prompt_tokens)
    _print_metrics(
        usage=usage,
        ttft=ttft,
        total_time=total_time,
        content="".join(content_parts),
        reasoning="".join(reasoning_parts),
        tool_calls=list(tool_calls.values()),
        debug_info=debug_info,
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
        choices=[SimpleNamespace(index=0, message=message, finish_reason=finish_reason)],
        usage=usage,
        ttft=ttft,
        total_time=total_time,
        decode_tokens_per_second=decode_tokens_per_second,
        cached_tokens=cached_tokens,
        cache_text=cache_text,
    )


def _print_metrics(
    *,
    usage,
    ttft: float,
    total_time: float,
    content: str,
    reasoning: str,
    tool_calls: list[dict],
    debug_info: dict,
) -> None:
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    speed = (
        completion_tokens / max(total_time - ttft, 1e-6)
        if isinstance(completion_tokens, int) and completion_tokens > 0 and total_time > ttft
        else 0.0
    )
    cached_tokens = _cached_tokens_from_chat_usage(usage)
    cache_text = _format_cache(cached_tokens, prompt_tokens)
    print(
        "\n[LLM] "
        f"TTFT={ttft:.3f}s time={total_time:.3f}s decode={speed:.2f} tok/s "
        f"in={prompt_tokens if prompt_tokens is not None else 'n/a'} "
        f"out={completion_tokens if completion_tokens is not None else 'n/a'} "
        f"cache={cache_text}",
        flush=True,
    )
    if not content and not reasoning and not tool_calls and completion_tokens:
        print(
            "\n[DeepSeek LLM Warning] completion tokens were produced, but no visible content, reasoning_content, or tool_calls were streamed. "
            f"request={debug_info}",
            flush=True,
        )


def _cached_tokens_from_chat_usage(usage) -> int | None:
    details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
    if details is None:
        return None
    if isinstance(details, dict):
        value = details.get("cached_tokens")
    else:
        value = getattr(details, "cached_tokens", None)
    return value if isinstance(value, int) else None


def _format_cache(cached_tokens: int | None, prompt_tokens: int | None) -> str:
    if cached_tokens is None:
        return "n/a"
    if isinstance(prompt_tokens, int) and prompt_tokens > 0:
        return f"{cached_tokens}/{prompt_tokens}={cached_tokens / prompt_tokens:.1%}"
    return str(cached_tokens)


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


def _normalize_reasoning_effort(value: str) -> str:
    effort = (value or "high").strip().lower()
    if effort in {"max", "xhigh"}:
        return "max"
    if effort in {"low", "medium", "high"}:
        return "high"
    return "high"


def _thinking_mode() -> str:
    value = (
        os.getenv("ONLYTEAM_THINKING")
        or os.getenv("DEEPSEEK_THINKING")
        or os.getenv("LOCAL_LLM_ENABLE_THINKING")
        or "enabled"
    ).strip().lower()
    if value in {"0", "false", "no", "off", "disabled", "disable"}:
        return "disabled"
    return "enabled"


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
                "\n[DeepSeek LLM Warning] "
                f"no tool calls returned; captured assistant content chars={len(text)} "
                f"preview={preview}",
                flush=True,
            )

    yield SimpleNamespace(
        type="response.completed",
        response=SimpleNamespace(
            id=getattr(completion, "id", f"resp_{uuid.uuid4().hex}"),
            output=output_items,
            usage=_to_responses_usage(
                getattr(completion, "usage", None),
                ttft=getattr(completion, "ttft", None),
                total_time=getattr(completion, "total_time", None),
                decode_tokens_per_second=getattr(completion, "decode_tokens_per_second", None),
                cached_tokens=getattr(completion, "cached_tokens", None),
                cache_text=getattr(completion, "cache_text", None),
            ),
            finish_reason=getattr(choice, "finish_reason", None),
        ),
    )


def _to_responses_usage(
    usage,
    *,
    ttft: float | None = None,
    total_time: float | None = None,
    decode_tokens_per_second: float | None = None,
    cached_tokens: int | None = None,
    cache_text: str | None = None,
):
    if usage is None:
        return None
    return SimpleNamespace(
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        total_tokens=getattr(usage, "total_tokens", 0) or 0,
        ttft=ttft,
        total_time=total_time,
        decode_tokens_per_second=decode_tokens_per_second,
        cached_tokens=cached_tokens,
        cache_text=cache_text,
    )


__all__ = ["deepseek_llm"]
