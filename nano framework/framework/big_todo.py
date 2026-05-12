"""Big-todo state and tools for Codex-style single-agent flows."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .mem_paths import MEM_ROOT, run_dir
from .tools import tool

MIN_STEPS = 1
MAX_STEPS = 12
PRE_BIG_TODO_ALLOWED = {
    "chat",
    "end_decide",
    "list_directory",
    "glob_files",
    "grep_search",
    "read_file",
    "read_many",
    "run_python",
    "run_powershell",
    "todo",
}


def big_todo_path(run_id: str, node_name: str) -> Path:
    safe = node_name.replace("/", "_").replace("\\", "_")
    return run_dir(run_id, MEM_ROOT) / f"{safe}.big_todo.json"


def load_big_todo(run_id: str | None, node_name: str | None) -> dict[str, Any]:
    if not run_id or not node_name:
        return _empty_state()
    path = big_todo_path(str(run_id), str(node_name))
    if not path.exists():
        return _empty_state()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _empty_state()
    return _normalize_state(payload if isinstance(payload, dict) else {})


def save_big_todo(run_id: str | None, node_name: str | None, state: dict[str, Any]) -> None:
    if not run_id or not node_name:
        return
    path = big_todo_path(str(run_id), str(node_name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_normalize_state(state), ensure_ascii=False, indent=2), encoding="utf-8")


def has_big_todo(run_id: str | None, node_name: str | None) -> bool:
    state = load_big_todo(run_id, node_name)
    return bool(state.get("ready") and state.get("goal") and state.get("steps"))


def render_big_todo_for_prompt(run_id: str | None, node_name: str | None) -> str:
    return "BIG_TODO\n" + render_big_todo(load_big_todo(run_id, node_name))


def render_big_todo(state: dict[str, Any]) -> str:
    state = _normalize_state(state)
    if not state["ready"]:
        return "big todo is not defined"

    lines = [f"goal: {state['goal']}", ""]
    for index, item in enumerate(state["steps"], start=1):
        mark = "x" if item.get("done") else " "
        lines.append(f"{index}. [{mark}] {item['step']}")
        lines.append(f"   check: {item['check']}")
        tips = str(item.get("tips") or "").strip()
        if tips:
            lines.append("   tips:")
            lines.extend(f"     {line}" for line in tips.splitlines())
    return "\n".join(lines).rstrip()


def pre_tool_gate(tool_name: str, run_id: str | None, node_name: str | None) -> str | None:
    if has_big_todo(run_id, node_name):
        return None
    if tool_name in PRE_BIG_TODO_ALLOWED:
        return None
    return (
        "TODO_BLOCKED: big todo is not defined. First inspect minimal context, then call "
        "todo(action='set', payload_json={\"goal\":\"...\",\"steps\":[{\"step\":\"...\",\"check\":\"...\"}]})"
    )


def final_gate(run_id: str | None, node_name: str | None, workspace: str | None) -> tuple[bool, str]:
    del workspace
    state = load_big_todo(run_id, node_name)
    if not state["ready"]:
        return False, "FINAL_BLOCKED:\n- big todo is not defined"

    unfinished = [
        f"{index}. {item['step']}"
        for index, item in enumerate(state["steps"], start=1)
        if not item.get("done")
    ]
    if unfinished:
        return False, "FINAL_BLOCKED:\n" + "\n".join(f"- step not done: {item}" for item in unfinished)
    return True, "FINAL_OK"


@tool(
    description=(
        "Manage the one and only big todo. "
        "Use action='set' exactly once after minimal context inspection to define goal and steps. "
        "Use action='done' with a 1-based step number plus tips to mark that step done. "
        "tips should capture key insights, earlier mistakes, the adopted approach and its effect, "
        "and what this implies for the next decision. "
        "Do not rewrite the whole todo just to update one completion mark."
    ),
    params={
        "action": ("string", "set or done"),
        "payload_json": (
            "string",
            "For set: {\"goal\":\"...\",\"steps\":[{\"step\":\"...\",\"check\":\"...\"}]}. "
            "For done: {\"step\":1,\"tips\":\"key insights; earlier mistakes; current approach and effect; next decision\"}.",
        ),
    },
)
def todo(action: str, payload_json: str = "{}", runtime_context: dict[str, Any] | None = None) -> str:
    context = runtime_context or {}
    action = (action or "").strip().lower()
    try:
        payload = json.loads(payload_json or "{}")
    except json.JSONDecodeError:
        return "TODO_ERROR: payload_json must be a JSON object"
    if not isinstance(payload, dict):
        return "TODO_ERROR: payload_json must be a JSON object"

    if action == "set":
        return _set_big_todo(payload, context.get("run_id"), context.get("node_name"))
    if action == "done":
        return _mark_step_done(payload, context.get("run_id"), context.get("node_name"))
    return "TODO_ERROR: action must be set or done"


@tool(
    description="End the current session. Use this only when the work is complete or the user explicitly wants to stop.",
    params={
        "reason": ("string", "Short reason shown in the route display."),
    },
)
def end_decide(reason: str, runtime_context: dict[str, Any] | None = None) -> str:
    context = runtime_context or {}
    ok, message = final_gate(
        run_id=context.get("run_id"),
        node_name=context.get("node_name"),
        workspace=context.get("workspace"),
    )
    if not ok:
        return message
    return f"DECIDE:end:{reason or ''}"


def _set_big_todo(payload: dict[str, Any], run_id: str | None, node_name: str | None) -> str:
    if has_big_todo(run_id, node_name):
        return "TODO_ERROR: big todo is already defined; do not rewrite it"
    goal = str(payload.get("goal") or "").strip()
    steps, error = _normalize_steps_payload(payload)
    if not goal:
        return "TODO_ERROR: goal is required"
    if error:
        return error

    state = {
        "ready": True,
        "goal": goal,
        "steps": steps,
        "updated_at": _now(),
    }
    save_big_todo(run_id, node_name, state)
    return "TODO_OK\n" + render_big_todo(state)


def _mark_step_done(payload: dict[str, Any], run_id: str | None, node_name: str | None) -> str:
    state = load_big_todo(run_id, node_name)
    if not state["ready"]:
        return "TODO_ERROR: big todo is not defined"

    raw_step = payload.get("step")
    try:
        step_index = int(raw_step)
    except (TypeError, ValueError):
        return "TODO_ERROR: step must be a 1-based integer"

    if step_index < 1 or step_index > len(state["steps"]):
        return f"TODO_ERROR: step must be between 1 and {len(state['steps'])}"

    tips = str(payload.get("tips") or "").strip()
    if not tips:
        return (
            "TODO_ERROR: tips is required when marking a step done; include key insights, earlier mistakes, "
            "the adopted approach and its effect, and what this changes for the next decision"
        )

    state["steps"][step_index - 1]["done"] = True
    state["steps"][step_index - 1]["tips"] = tips
    state["updated_at"] = _now()
    save_big_todo(run_id, node_name, state)
    return "TODO_OK\n" + render_big_todo(state)


def _normalize_state(payload: dict[str, Any]) -> dict[str, Any]:
    goal = str(payload.get("goal") or "").strip()
    steps, _ = _normalize_steps_payload(payload)
    return {
        "ready": bool(payload.get("ready") and goal and steps),
        "goal": goal,
        "steps": steps,
        "updated_at": str(payload.get("updated_at") or _now()),
    }


def _normalize_steps_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return [], "TODO_ERROR: steps must be a JSON array"
    if not (MIN_STEPS <= len(raw_steps) <= MAX_STEPS):
        return [], f"TODO_ERROR: steps must contain {MIN_STEPS}-{MAX_STEPS} items"

    steps: list[dict[str, Any]] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            return [], "TODO_ERROR: each step must be an object"
        step = str(raw.get("step") or "").strip()
        check = str(raw.get("check") or "").strip()
        if not step:
            return [], "TODO_ERROR: step text is required"
        if not check:
            return [], "TODO_ERROR: step check is required"
        steps.append(
            {
                "step": step,
                "check": check,
                "done": bool(raw.get("done")),
                "tips": str(raw.get("tips") or "").strip(),
            }
        )
    return steps, None


def _empty_state() -> dict[str, Any]:
    return {"ready": False, "goal": "", "steps": [], "updated_at": _now()}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "big_todo_path",
    "end_decide",
    "final_gate",
    "has_big_todo",
    "load_big_todo",
    "pre_tool_gate",
    "todo",
    "render_big_todo",
    "render_big_todo_for_prompt",
    "save_big_todo",
]
