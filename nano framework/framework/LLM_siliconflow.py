"""SiliconFlow adapter.

Provides a Responses-style streaming interface for systems that currently
consume OpenAI Responses API events.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

from openai import OpenAI


class _EventItem(SimpleNamespace):
    def to_dict(self):
        return dict(self.__dict__)


def _load_env() -> None:
    """Load the first .env file found near the project."""
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
                os.environ[key.strip()] = value.strip()
        return


_load_env()


def siliconflow_llm(
    model: str = "Qwen/Qwen3-32B",
    reasoning_effort: str = "",
    reasoning_summary: str = "",
    tool_choice: str = "required",
):
    """Create one SiliconFlow caller with Responses-like streamed events."""
    del reasoning_effort, reasoning_summary

    client = OpenAI(
        api_key=os.getenv("SILICONFLOW_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1"),
    )

    def llm(
        input_items: list[dict],
        tools: list[dict] | None = None,
        instructions: str = "",
    ):
        messages = _to_chat_messages(input_items=input_items, instructions=instructions)
        normalized_tools = _to_siliconflow_tools(tools)

        kwargs = {
            "model": model,
            "messages": messages,
            "stream": False,
        }

        if normalized_tools:
            kwargs["tools"] = normalized_tools
            kwargs["tool_choice"] = tool_choice

        completion = client.chat.completions.create(**kwargs)
        return _emit_responses_style_events(completion)

    return llm


def _to_chat_messages(input_items: list[dict], instructions: str) -> list[dict]:
    messages: list[dict] = []

    if instructions:
        messages.append({"role": "system", "content": instructions})

    for item in input_items:
        role = item.get("role")
        item_type = item.get("type")

        if role in {"user", "assistant", "system"}:
            content = item.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") in {"input_text", "output_text", "text"}:
                        text_parts.append(str(block.get("text", "")))
                content = "\n".join(part for part in text_parts if part)
            messages.append({"role": role, "content": str(content)})
            continue

        if item_type == "function_call":
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}",
                            "type": "function",
                            "function": {
                                "name": item.get("name", "unknown_tool"),
                                "arguments": item.get("arguments", "{}") or "{}",
                            },
                        }
                    ],
                }
            )
            continue

        if item_type == "function_call_output":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": item.get("call_id", ""),
                    "content": str(item.get("output", "")),
                }
            )

    return messages


def _to_siliconflow_tools(tools: list[dict] | None) -> list[dict]:
    if not tools:
        return []

    normalized: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue

        # Accept both:
        # 1) {"type":"function","function":{"name":...,"parameters":...}}
        # 2) {"type":"function","name":...,"parameters":...} (Responses-style)
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
        if not name:
            continue
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

    tool_calls = getattr(message, "tool_calls", None) or []
    for tc in tool_calls:
        func = tc.function
        item_id = f"fc_{uuid.uuid4().hex}"
        call_id = getattr(tc, "id", None) or f"call_{uuid.uuid4().hex}"
        name = getattr(func, "name", "unknown_tool")
        arguments = getattr(func, "arguments", "{}") or "{}"

        item = _EventItem(
            type="function_call",
            id=item_id,
            call_id=call_id,
            name=name,
            arguments=arguments,
            status="completed",
        )
        output_items.append(item)

        yield SimpleNamespace(type="response.output_item.added", item=item)
        yield SimpleNamespace(type="response.function_call_arguments.delta", item_id=item_id, delta=arguments)
        yield SimpleNamespace(type="response.output_item.done", item=item)

    if not tool_calls:
        text = message.content or ""
        if isinstance(text, list):
            text = "\n".join(
                str(part.text)
                for part in text
                if hasattr(part, "text") and part.text
            )

        msg_item = _EventItem(
            type="message",
            id=f"msg_{uuid.uuid4().hex}",
            role="assistant",
            content=[{"type": "output_text", "text": str(text)}],
            status="completed",
        )
        output_items.append(msg_item)
        yield SimpleNamespace(type="response.output_item.done", item=msg_item)

    response_obj = SimpleNamespace(
        id=getattr(completion, "id", f"resp_{uuid.uuid4().hex}"),
        output=output_items,
        usage=getattr(completion, "usage", None),
    )
    yield SimpleNamespace(type="response.completed", response=response_obj)


def _safe_json_dumps(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return "{}"


__all__ = ["siliconflow_llm"]
