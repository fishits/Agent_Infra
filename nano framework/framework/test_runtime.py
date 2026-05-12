from __future__ import annotations

import unittest

from framework.runtime import _format_tool_result_preview


class RuntimePreviewTests(unittest.TestCase):
    def test_tool_result_preview_uses_range_style(self) -> None:
        text = "# file: main.py\n# lines: 1-23/330\n" + ("x" * 6000)

        rendered = _format_tool_result_preview(text)

        self.assertIn("chars: 4000/", rendered)
        self.assertIn("lines: 1-23/330", rendered)
        self.assertNotIn("truncated", rendered.lower())
        self.assertNotIn("省略", rendered)


if __name__ == "__main__":
    unittest.main()
