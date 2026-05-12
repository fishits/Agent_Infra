from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from openai import APIConnectionError

from framework.deepseek_llm import _emit_responses_style_events, _project_input_items, _to_responses_usage, deepseek_llm


class DeepSeekThinkingTests(unittest.TestCase):
    def test_project_input_items_replays_tool_call_reasoning_content(self) -> None:
        messages = _project_input_items(
            [
                {"role": "user", "content": "start"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path":"a.py","start_line":1,"end_line":10}',
                    "thinking": "I need to inspect the file before editing.",
                },
                {"type": "function_call_output", "call_id": "call_1", "output": "file content"},
            ],
            "system prompt",
        )

        assistant = messages[2]
        self.assertEqual(assistant["role"], "assistant")
        self.assertEqual(assistant["reasoning_content"], "I need to inspect the file before editing.")
        self.assertEqual(assistant["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[3]["role"], "tool")
        self.assertEqual(messages[3]["tool_call_id"], "call_1")

    def test_deepseek_request_sets_thinking_and_reasoning_effort(self) -> None:
        captured: dict = {}

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.base_url = kwargs.get("base_url")
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch("framework.deepseek_llm.OpenAI", FakeOpenAI):
            llm = deepseek_llm(model="deepseek-v4-pro", reasoning_effort="xhigh")
            list(llm(input_items=[{"role": "user", "content": "hello"}], tools=[], instructions=""))

        self.assertEqual(captured["reasoning_effort"], "max")
        self.assertEqual(captured["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertNotIn("temperature", captured)
        self.assertNotIn("top_p", captured)

    def test_deepseek_retries_api_connection_error(self) -> None:
        attempts = 0

        class FakeCompletions:
            def create(self, **kwargs):
                nonlocal attempts
                attempts += 1
                if attempts < 3:
                    raise APIConnectionError(request=None)
                return []

        class FakeOpenAI:
            def __init__(self, **kwargs):
                self.base_url = kwargs.get("base_url")
                self.chat = SimpleNamespace(completions=FakeCompletions())

        with patch("framework.deepseek_llm.OpenAI", FakeOpenAI), patch("framework.deepseek_llm.time.sleep"):
            llm = deepseek_llm(model="deepseek-v4-pro", reasoning_effort="xhigh")
            list(llm(input_items=[{"role": "user", "content": "hello"}], tools=[], instructions=""))

        self.assertEqual(attempts, 3)

    def test_emit_events_keeps_assistant_text_and_finish_reason_without_tool_calls(self) -> None:
        completion = SimpleNamespace(
            id="resp_1",
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(
                        content="partial answer",
                        reasoning_content=None,
                        tool_calls=None,
                    ),
                )
            ],
            usage=None,
        )

        events = list(_emit_responses_style_events(completion))

        response_items = [event.item for event in events if event.type == "response.output_item.done"]
        self.assertEqual(len(response_items), 1)
        self.assertEqual(response_items[0].role, "assistant")
        self.assertEqual(response_items[0].content, "partial answer")
        completed = [event for event in events if event.type == "response.completed"][0]
        self.assertEqual(completed.response.finish_reason, "length")

    def test_to_responses_usage_keeps_latency_and_cache_metrics(self) -> None:
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3, total_tokens=15)

        converted = _to_responses_usage(
            usage,
            ttft=1.25,
            total_time=2.5,
            decode_tokens_per_second=4.0,
            cached_tokens=8,
            cache_text="8/12=66.7%",
        )

        self.assertEqual(converted.input_tokens, 12)
        self.assertEqual(converted.output_tokens, 3)
        self.assertEqual(converted.total_tokens, 15)
        self.assertEqual(converted.ttft, 1.25)
        self.assertEqual(converted.total_time, 2.5)
        self.assertEqual(converted.decode_tokens_per_second, 4.0)
        self.assertEqual(converted.cached_tokens, 8)
        self.assertEqual(converted.cache_text, "8/12=66.7%")


if __name__ == "__main__":
    unittest.main()
