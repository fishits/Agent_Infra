from __future__ import annotations

import unittest

from framework.prompts import build_run_context_prompt


class PromptContextTests(unittest.TestCase):
    def test_build_run_context_prompt_prefers_state_workspace(self) -> None:
        rendered = build_run_context_prompt(
            base_prompt="base",
            graph_state={
                "run_id": "run_1",
                "workspace": r"C:\Users\13090\Desktop\input",
                "root_user_message": "hello",
            },
            node_name="onlyteam_codex",
            workspace=r"C:\Users\13090\Desktop\FrameWork_light",
        )

        self.assertIn(r"- workspace: C:\Users\13090\Desktop\input", rendered)
        self.assertNotIn(r"- workspace: C:\Users\13090\Desktop\FrameWork_light", rendered)


if __name__ == "__main__":
    unittest.main()
