"""OnlyTeam control tools.

These tools are product-layer controls. They intentionally expose only
phase switching and user clarification.
"""

from __future__ import annotations

from typing import Any

from framework.tools import tool

from OnlyTeam.phase_runtime.acceptance_state import preflight_action
from OnlyTeam.phase_runtime.phases import END_PHASE, known_phase_names, next_phases


PHASE_TARGETS = known_phase_names()


@tool(
    description=(
        "Switch the OnlyTeam work mode. This keeps the same shared conversation history; "
        "it only updates state['phase']."
    ),
    params={
        "target": ("string", "Next phase: task_setup, data_profile, experiment_loop, final_report, or end"),
        "reason": ("string", "Short reason shown in the CLI route display. It is not injected as handoff content."),
    },
)
def decide(target: str, reason: str, runtime_context: dict[str, Any] | None = None) -> str:
    target = (target or "").strip()
    context = runtime_context or {}
    phase = str(context.get("phase") or context.get("current_phase") or "")
    if target not in PHASE_TARGETS:
        allowed = ", ".join(sorted(PHASE_TARGETS))
        return f"DECIDE_ERROR: target must be one of: {allowed}"
    if not phase:
        return "DECIDE_ERROR: runtime_context.phase is required for OnlyTeam mode switch"
    if phase == END_PHASE:
        return "DECIDE_ERROR: current phase is already end"
    allowed_targets = set(next_phases(phase))
    if target not in allowed_targets:
        allowed = ", ".join(sorted(allowed_targets))
        return f"DECIDE_ERROR: current phase {phase} can only route to: {allowed}"
    return f"DECIDE:{target}:{reason or ''}"


@tool(
    description="Ask the user for one concise clarification and pause the current phase.",
    params={
        "message": ("string", "Message shown to the user."),
    },
)
def chat(message: str) -> str:
    return f"CHAT:{message}"


@tool(
    description=(
        "Manage the OnlyTeam big todo before and during work. "
        "Use action='set' once after minimal context inspection to define goal and steps. "
        "Use action='check' with a 1-based step number to mark that step done. "
        "Use action='status' to read the current big todo without rewriting it."
    ),
    params={
        "action": ("string", "set, check, or status"),
        "payload_json": (
            "string",
            "For set: {\"goal\":\"...\",\"steps\":[{\"step\":\"...\",\"check\":\"...\"}]}. "
            "For check: {\"step\":1}. For status: {}.",
        ),
    },
)
def preflight(action: str, payload_json: str = "{}", runtime_context: dict[str, Any] | None = None) -> str:
    context = runtime_context or {}
    return preflight_action(
        action=action,
        payload_json=payload_json,
        run_id=context.get("run_id"),
        node_name=context.get("node_name"),
        workspace=context.get("workspace"),
    )


__all__ = ["PHASE_TARGETS", "chat", "decide", "preflight"]
