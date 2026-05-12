from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from framework.io_tools import MAX_READ_CHARS, write_file as real_write_file
from framework.tools import call_tool, tool


@tool(
    description="测试写入工具",
    params={
        "path": ("string", "目标路径"),
        "content": ("string", "写入内容"),
    },
)
def write_file(path: str, content: str) -> str:
    return f"{path}:{content}"


class ToolSanitizationTests(unittest.TestCase):
    def test_call_tool_drops_internal_history_keys(self) -> None:
        result = call_tool(
            [write_file],
            "write_file",
            {
                "path": "main.py",
                "content": "print(1)",
                "历史工具参数": "坏字段",
                "_omitted": True,
            },
        )

        self.assertEqual(result, "main.py:print(1)")

    def test_call_tool_rejects_history_placeholder_for_write_tools(self) -> None:
        result = call_tool(
            [write_file],
            "write_file",
            {
                "path": "main.py",
                "content": "历史 content 已省略。字符数=3231",
            },
        )

        self.assertIn("拒绝执行写入", result)

    def test_real_write_file_truncates_and_writes_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "a" * (MAX_READ_CHARS + 25)
            result = real_write_file(
                "out.txt",
                content,
                runtime_context={"cwd": tmpdir},
            )
            written = Path(tmpdir, "out.txt").read_text(encoding="utf-8")

        self.assertEqual(len(written), MAX_READ_CHARS)
        self.assertEqual(written, content[:MAX_READ_CHARS])
        self.assertIn("# path: out.txt", result)
        self.assertIn("单次输出过长已被截断", result)
        self.assertIn("请使用 apply_patch、append_file", result)


if __name__ == "__main__":
    unittest.main()
