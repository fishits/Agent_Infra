"""Local model API adapter for SGLang.

Adapted to use Chat Completions API and convert responses to Responses API format.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterator, Any

from openai import OpenAI, InternalServerError


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


class ResponseItem:
    """Mock response item to match Responses API format."""
    def __init__(self, item_type: str, content: str = "", **kwargs):
        self.type = item_type
        self.content = content
        for key, value in kwargs.items():
            setattr(self, key, value)


class ResponseEvent:
    """Mock response event to match Responses API format."""
    def __init__(self, event_type: str, **kwargs):
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)


def _convert_chat_stream_to_responses(stream: Iterator[Any], response_id: str) -> Iterator[ResponseEvent]:
    """Convert Chat Completions stream to Responses API format."""
    tool_calls_buffer = {}
    content_buffer = ""
    
    for chunk in stream:
        if not chunk.choices:
            continue
            
        choice = chunk.choices[0]
        delta = choice.delta
        
        # Handle content delta
        if delta.content:
            content_buffer += delta.content
            # Framework doesn't seem to use content deltas, skip for now
        
        # Handle tool calls
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                index = tool_call.index
                
                if index not in tool_calls_buffer:
                    # New tool call
                    tool_calls_buffer[index] = {
                        "id": tool_call.id or f"call_{index}",
                        "name": tool_call.function.name if tool_call.function else "",
                        "arguments": "",
                    }
                    
                    # Emit tool_call_start event
                    item = ResponseItem(
                        "function_call",
                        id=tool_calls_buffer[index]["id"],
                        call_id=tool_calls_buffer[index]["id"],
                        name=tool_calls_buffer[index]["name"],
                    )
                    yield ResponseEvent(
                        "response.output_item.added",
                        item=item,
                    )
                
                # Accumulate arguments
                if tool_call.function and tool_call.function.arguments:
                    tool_calls_buffer[index]["arguments"] += tool_call.function.arguments
                    
                    # Emit arguments delta
                    yield ResponseEvent(
                        "response.function_call_arguments.delta",
                        item_id=tool_calls_buffer[index]["id"],
                        delta=tool_call.function.arguments,
                    )
        
        # Handle finish
        if choice.finish_reason:
            # Emit completed tool calls
            for index, tool_call_data in tool_calls_buffer.items():
                item = ResponseItem(
                    "function_call",
                    id=tool_call_data["id"],
                    call_id=tool_call_data["id"],
                    name=tool_call_data["name"],
                    arguments=tool_call_data["arguments"],
                )
                yield ResponseEvent(
                    "response.output_item.done",
                    item=item,
                )
            
            # Emit response completed
            class MockResponse:
                def __init__(self):
                    self.id = response_id
                    self.output = []
                    self.usage = None
                    
            yield ResponseEvent(
                "response.completed",
                response=MockResponse(),
            )


def local_llm(
    model: str = "qwen3-coder-next",
    reasoning_effort: str = "",
    reasoning_summary: str = "",
    tool_choice: str = "auto",
):
    """Create one local model API caller for SGLang using Chat Completions API.
    
    This adapter converts between Responses API format (used by framework) and 
    Chat Completions API format (supported by SGLang).
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    def llm(
        input_items: list[dict],
        tools: list[dict] | None = None,
        instructions: str = "",
    ):
        # Convert Responses API format to Chat Completions API format
        messages = []
        
        # Add system message if instructions provided
        if instructions:
            messages.append({"role": "system", "content": instructions})
        
        # Add input messages
        messages.extend(input_items)
        
        # Build request kwargs
        kwargs = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        
        # Convert tools format: framework uses old OpenAI format, SGLang needs new format
        if tools:
            converted_tools = []
            for tool in tools:
                if "function" in tool:
                    converted_tools.append(tool)
                else:
                    converted_tools.append({
                        "type": "function",
                        "function": {
                            "name": tool.get("name"),
                            "description": tool.get("description"),
                            "parameters": tool.get("parameters"),
                        }
                    })
            kwargs["tools"] = converted_tools
            kwargs["tool_choice"] = tool_choice

        # Debug info (optional, can be removed for cleaner output)
        # debug_info = {
        #     "model": kwargs.get("model"),
        #     "messages_count": len(kwargs.get("messages", [])),
        #     "system_prompt_length": len(instructions) if instructions else 0,
        #     "tools_count": len(kwargs.get("tools", [])) if kwargs.get("tools") else 0,
        #     "tool_choice": kwargs.get("tool_choice"),
        #     "stream": kwargs.get("stream"),
        # }
        # print(f"\n[Local LLM] Using Chat Completions API")
        # print(f"[Local LLM Debug] Request params: {debug_info}")

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                chat_stream = client.chat.completions.create(**kwargs)
                # Convert Chat Completions stream to Responses API format
                response_id = f"resp_{int(time.time() * 1000)}"
                return _convert_chat_stream_to_responses(chat_stream, response_id)
            except InternalServerError as e:
                if attempt == max_retries:
                    print(f"\n[Local LLM Error] 鏈€缁堝け璐ワ紝璇锋眰鍙傛暟: {debug_info}")
                    raise
                wait = 2 ** attempt
                print(f"\n[Local LLM] 鏈嶅姟绔敊璇?(attempt {attempt}/{max_retries}): {e.message} 鈥?绛夊緟 {wait}s 鍚庨噸璇?..")
                time.sleep(wait)
            except Exception as e:
                print(f"\n[Local LLM Error] 鏈鏈熺殑閿欒: {type(e).__name__}: {e}")
                print(f"[Local LLM Error] 璇锋眰鍙傛暟: {debug_info}")
                raise

    return llm


__all__ = ["local_llm"]
