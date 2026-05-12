"""Responses API adapter.

This file only loads config and sends one Responses API request.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

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
                os.environ.setdefault(key.strip(), value.strip())
        return


_load_env()


def openai_llm(
    model: str = "gpt-5.4",
    reasoning_effort: str = "low",
    reasoning_summary: str = "auto",
    tool_choice: str = "required",
):
    """Create one Responses API caller."""
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    def llm(
        input_items: list[dict],
        tools: list[dict] | None = None,
        instructions: str = "",
    ):
        kwargs = {
            "model": model,
            "input": input_items,
            "instructions": instructions or None,
            "stream": True,
            "store": True,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice

        reasoning = {}
        if reasoning_effort:
            reasoning["effort"] = reasoning_effort
        if reasoning_summary:
            reasoning["summary"] = reasoning_summary
        if reasoning:
            kwargs["reasoning"] = reasoning

        clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}

        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                return client.responses.create(**clean_kwargs)
            except InternalServerError as e:
                if attempt == max_retries:
                    raise
                wait = 2 ** attempt  # 2, 4, 8, 16 秒
                print(f"\n[LLM] 服务端错误 (attempt {attempt}/{max_retries}): {e.message} — 等待 {wait}s 后重试...")
                time.sleep(wait)

    return llm


__all__ = ["openai_llm"]
