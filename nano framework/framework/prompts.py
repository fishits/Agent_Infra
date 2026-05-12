"""Prompt definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

def coding_executor_prompt() -> str:
    return """You are the coding executor.
Rules:
1) Freeze evaluation policy before training.
2) Use PowerShell only. Do not use bash.
3) Keep one Python interpreter path fixed in each run.
4) For every experiment, output:
   - experiment_id
   - policy_version
   - coverage_count
   - rmse_improvement_vs_persistence_pct
   - mae_improvement_vs_persistence_pct
   - pearson_r
   - r2
   - artifact_paths
5) Escalate to boss only via:
   - SIGNOFF_PACKAGE (complete package)
   - DECISION_REQUIRED (single explicit decision)
"""


def boss_report_prompt() -> str:
    return """Prepare a boss-facing report using one schema only.

SIGNOFF_PACKAGE:
- policy_version
- policy_text
- final_test_metrics
- pass_fail_table
- artifacts (absolute paths)
- risks

DECISION_REQUIRED:
- decision_question
- options
- impact_if_option_a
- impact_if_option_b
- recommendation
"""


def joke_teller_prompt() -> str:
    """Backward-compatible demo prompt."""
    return "Tell one short joke, then use decide(target='joke_judge', content='...')."


def joke_judge_prompt() -> str:
    """Backward-compatible demo prompt."""
    return "Score the joke from 1-10 and ask user feedback via chat(...)."


def build_run_context_prompt(
    base_prompt: str,
    graph_state: dict[str, Any],
    node_name: str,
    workspace: str | Path,
) -> str:
    run_id = str(graph_state.get("run_id") or "")
    workspace_value = graph_state.get("workspace") or workspace
    return "\n\n".join(
        [
            base_prompt.strip(),
            "RUN_CONTEXT\n"
            f"- run_id: {run_id}\n"
            f"- workspace: {Path(str(workspace_value)).resolve()}\n"
            f"- memory_scope: {node_name}\n"
            f"- root_user_request: {graph_state.get('root_user_message') or '(not set)'}",
        ]
    )


def bind_run_context_prompt(base_prompt: str, node_name: str, workspace: str | Path):
    return lambda state: build_run_context_prompt(base_prompt, state, node_name, workspace)


__all__ = [
    "bind_run_context_prompt",
    "coding_executor_prompt",
    "boss_report_prompt",
    "build_run_context_prompt",
    "joke_judge_prompt",
    "joke_teller_prompt",
]
