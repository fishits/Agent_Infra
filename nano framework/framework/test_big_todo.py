from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from framework.big_todo import load_big_todo, render_big_todo, todo


class BigTodoTipsTests(unittest.TestCase):
    def test_done_requires_tips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_root = Path(tmpdir)
            with patch("framework.big_todo.MEM_ROOT", mem_root):
                runtime_context = {"run_id": "run_1", "node_name": "node_1"}
                todo(
                    "set",
                    '{"goal":"Ship feature","steps":[{"step":"Investigate","check":"Have root cause"}]}',
                    runtime_context=runtime_context,
                )

                result = todo("done", '{"step":1}', runtime_context=runtime_context)

        self.assertIn("TODO_ERROR: tips is required", result)

    def test_done_persists_and_renders_tips(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mem_root = Path(tmpdir)
            with patch("framework.big_todo.MEM_ROOT", mem_root):
                runtime_context = {"run_id": "run_1", "node_name": "node_1"}
                todo(
                    "set",
                    '{"goal":"Ship feature","steps":[{"step":"Investigate","check":"Have root cause"}]}',
                    runtime_context=runtime_context,
                )

                result = todo(
                    "done",
                    '{"step":1,"tips":"key insight\\nprevious mistake\\ncurrent approach works\\nnext: keep the stable prefix"}',
                    runtime_context=runtime_context,
                )
                state = load_big_todo("run_1", "node_1")

        self.assertEqual(state["steps"][0]["tips"], "key insight\nprevious mistake\ncurrent approach works\nnext: keep the stable prefix")
        self.assertIn("tips:", result)
        self.assertIn("previous mistake", render_big_todo(state))


if __name__ == "__main__":
    unittest.main()
