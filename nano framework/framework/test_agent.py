from __future__ import annotations

import unittest
from types import SimpleNamespace

from framework.agent import _build_usage_record, agent_loop


class AgentLoopThinkingTests(unittest.TestCase):
    def test_agent_loop_attaches_thinking_to_function_call_items(self) -> None:
        def fake_llm(input_items, tools=None, instructions=""):
            del input_items, tools, instructions
            item = SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="read_file",
                arguments='{"path":"a.py"}',
                status="completed",
            )
            yield SimpleNamespace(type="response.reasoning_summary_text.delta", delta="inspect first")
            yield SimpleNamespace(type="response.output_item.added", item=item)
            yield SimpleNamespace(type="response.function_call_arguments.delta", item_id="fc_1", delta='{"path":"a.py"}')
            yield SimpleNamespace(type="response.output_item.done", item=item)
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp_1", output=[item], usage=None),
            )

        node = {
            "llm": fake_llm,
            "tools": [],
            "prompt": "test",
            "edges": None,
            "run_id": "run",
            "name": "node",
            "graph_state": {},
        }

        response_items = [event["item"] for event in agent_loop(node, [{"role": "user", "content": "go"}]) if event["type"] == "response_item"]

        self.assertEqual(len(response_items), 1)
        self.assertEqual(getattr(response_items[0], "thinking", ""), "inspect first")

    def test_agent_loop_auto_continues_after_length_stop(self) -> None:
        calls: list[list[dict]] = []

        def fake_llm(input_items, tools=None, instructions=""):
            del tools, instructions
            calls.append(input_items)
            if len(calls) == 1:
                item = SimpleNamespace(type="message", role="assistant", content="partial")
                yield SimpleNamespace(type="response.output_item.done", item=item)
                yield SimpleNamespace(
                    type="response.completed",
                    response=SimpleNamespace(id="resp_1", output=[item], usage=None, finish_reason="length"),
                )
                return

            tool_item = SimpleNamespace(
                type="function_call",
                id="fc_1",
                call_id="call_1",
                name="read_file",
                arguments='{"path":"a.py"}',
                status="completed",
            )
            yield SimpleNamespace(type="response.output_item.added", item=tool_item)
            yield SimpleNamespace(type="response.function_call_arguments.delta", item_id="fc_1", delta='{"path":"a.py"}')
            yield SimpleNamespace(type="response.output_item.done", item=tool_item)
            yield SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(id="resp_2", output=[tool_item], usage=None, finish_reason="stop"),
            )

        node = {
            "llm": fake_llm,
            "tools": [],
            "prompt": "test",
            "edges": None,
            "run_id": "run",
            "name": "node",
            "graph_state": {},
        }

        events = list(agent_loop(node, [{"role": "user", "content": "go"}]))
        tool_calls = [event for event in events if event["type"] == "tool_call"]

        self.assertEqual(len(calls), 2)
        self.assertEqual(tool_calls[0]["name"], "read_file")
        self.assertEqual(calls[1][-2]["role"], "assistant")
        self.assertEqual(calls[1][-2]["content"], "partial")

    def test_build_usage_record_keeps_latency_and_cache_metrics(self) -> None:
        usage = SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            ttft=1.1,
            total_time=3.2,
            decode_tokens_per_second=9.5,
            cached_tokens=80,
            cache_text="80/100=80.0%",
        )

        record = _build_usage_record(usage, "resp_1")

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["input_tokens"], 100)
        self.assertEqual(record["output_tokens"], 20)
        self.assertEqual(record["total_tokens"], 120)
        self.assertEqual(record["cached_tokens"], 80)
        self.assertEqual(record["cache_text"], "80/100=80.0%")
        self.assertAlmostEqual(record["ttft"], 1.1)
        self.assertAlmostEqual(record["total_time"], 3.2)
        self.assertAlmostEqual(record["decode_tokens_per_second"], 9.5)


if __name__ == "__main__":
    unittest.main()
