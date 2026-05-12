"""DatasetTeam prompt definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from framework.todo_state import read_todo_state

from DatasetTeam.acceptance_state import render_big_todo_for_prompt
from DatasetTeam.phases import AGENT_NODE, START_PHASE, next_phases, phase_spec


BASE_DATASET_TEAM_PROMPT = """\
You are Codex running DatasetTeam inside the local workspace.
Your job is not generic research. Your job is to turn one OnlyTeam memory run into RL preference data that is safe to consume by OpenRLHF.

Core operating rules:
- Work phase by phase. Use one tool every assistant turn.
- Do not invent data that is not in the source memory file.
- Treat output compatibility as a hard requirement.
- Use chat(message=...) only for concise user communication.
- Use decide(target=..., reason=...) only when the current phase gate is actually satisfied.
- If a tool fails, inspect the failure and recover instead of repeating blindly.

Data discipline:
- A DecisionRecord must come from a real paired function_call + function_call_output.
- recent_user_message must be evidence-based from prior user records in the same memory file.
- Judge output must be strict JSON only.
- Strict OpenRLHF export must contain only chosen and rejected keys per line.
- If validation fails, do not route to end.
"""


STABLE_TOOL_POLICY = """\
Stable tool policy:
- All phases share one tool schema; runtime gates decide which tools are allowed.
- If a tool returns TOOL_BLOCKED, you used a tool that is not allowed in the current phase.
- If a tool returns PREFLIGHT_BLOCKED, define the big todo first or satisfy a phase gate.
- In final_report, validation must pass before end.
"""


def dataset_team_prompt(_: dict[str, Any] | None = None) -> str:
    return ""


def build_node_prompt(node: dict[str, Any], state: dict[str, Any]) -> str:
    tools = node.get("tools") or []
    tool_names = ", ".join(fn.__name__ for fn in tools)
    return "\n\n".join(
        [
            BASE_DATASET_TEAM_PROMPT.strip(),
            STABLE_TOOL_POLICY.strip(),
            _section("STABLE_TOOL_SCHEMA", [f"available_tools: {tool_names}"]),
        ]
    )


def build_dynamic_context_items(node: dict[str, Any], state: dict[str, Any]) -> list[dict[str, str]]:
    phase = str(state.get("phase") or START_PHASE)
    workspace = Path(str(state.get("workspace") or Path.cwd())).resolve()
    run_id = str(state.get("run_id") or "")
    node_name = str(state.get("current_node") or node.get("name") or AGENT_NODE)
    content = "\n\n".join(
        [
            _section(
                "RUN_CONTEXT",
                [
                    f"run_id: {run_id}",
                    f"workspace: {workspace}",
                    f"output_dir: {state.get('output_dir') or '(not set)'}",
                    f"source_run_id: {state.get('source_run_id') or '(not set)'}",
                    f"judge_model: {state.get('judge_model') or '(not set)'}",
                    f"judge_reasoning_effort: {state.get('judge_reasoning_effort') or '(not set)'}",
                    f"compat_mode: {state.get('compat_mode') or '(not set)'}",
                    f"root_user_request: {state.get('root_user_message') or '(not set)'}",
                ],
            ),
            _phase_card(phase),
            render_big_todo_for_prompt(run_id, node_name),
            _small_todo_card(run_id, node_name),
        ]
    )
    return [{"role": "user", "content": content}]


def _phase_card(phase: str) -> str:
    spec = phase_spec(phase)
    lines = [
        "CURRENT_PHASE_CARD",
        f"- current_phase: {phase}",
        f"- objective: {spec['objective']}",
        f"- allowed_actions: {_join_items(spec['allowed_actions'])}",
        f"- acceptance_gate: {_join_items(spec['acceptance_gate'])}",
        f"- runtime_allowed_tools: {', '.join(str(name) for name in spec['allowed_tools'])}",
        f"- valid_next_phases: {', '.join(next_phases(phase))}",
        "- decide reason is for route display only; do not put large handoff payloads in it.",
    ]
    return "\n".join(lines)


def _small_todo_card(run_id: str, node_name: str) -> str:
    todos = read_todo_state(run_id, node_name)
    if not todos:
        return "SMALL_TODO_STATE\nsmall todo is empty"
    lines = ["SMALL_TODO_STATE"]
    for item in todos:
        lines.append(f"- [{item['status']}] {item['content']}")
    return "\n".join(lines)


def _section(title: str, items: list[str]) -> str:
    return "\n".join([title, *[f"- {item}" for item in items]])


def _join_items(value: object) -> str:
    if isinstance(value, list):
        return " | ".join(str(item) for item in value)
    return str(value)


__all__ = [
    "BASE_DATASET_TEAM_PROMPT",
    "STABLE_TOOL_POLICY",
    "build_dynamic_context_items",
    "build_node_prompt",
    "dataset_team_prompt",
]
