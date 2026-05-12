"""OnlyTeam minimal big-todo state.

Big todo is a readable task contract, not a runtime log. It stores only:
goal, steps, each step's check text, and whether the step is done.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PRE_BIG_TODO_ALLOWED = {
    "chat",
    "decide",
    "list_directory",
    "glob_files",
    "grep_search",
    "read_file",
    "read_many",
    "run_python",
    "preflight",
    "todo_read",
    "todo_write",
}
MIN_STEPS = 1
MAX_STEPS = 12


def big_todo_path(run_id: str, node_name: str) -> Path:
    safe = node_name.replace("/", "_").replace("\\", "_")
    root = Path(__file__).resolve().parents[2] / ".mem"
    return root / run_id / f"{safe}.big_todo.json"


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


def preflight_action(
    *,
    action: str,
    payload_json: str,
    run_id: str | None,
    node_name: str | None,
    workspace: str | None,
) -> str:
    del workspace
    action = (action or "").strip().lower()
    if action == "status":
        return render_big_todo(load_big_todo(run_id, node_name))

    try:
        payload = json.loads(payload_json or "{}")
    except json.JSONDecodeError:
        return "PREFLIGHT_ERROR: payload_json must be a JSON object"
    if not isinstance(payload, dict):
        return "PREFLIGHT_ERROR: payload_json must be a JSON object"

    if action == "set":
        return _set_big_todo(payload, run_id, node_name)
    if action == "check":
        return _mark_step_done(payload, run_id, node_name)
    return "PREFLIGHT_ERROR: action must be set, check, or status"


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
        lines.append(f"   验收: {item['check']}")
    return "\n".join(lines).rstrip()


def record_tool_fact(
    *,
    run_id: str | None,
    node_name: str | None,
    workspace: str | None,
    tool_name: str,
    args: dict[str, Any],
    result: str,
) -> None:
    del run_id, node_name, workspace, tool_name, args, result
    return None


def pre_tool_gate(tool_name: str, phase: str, run_id: str | None, node_name: str | None) -> str | None:
    if has_big_todo(run_id, node_name):
        return None
    if tool_name in PRE_BIG_TODO_ALLOWED:
        if tool_name == "decide" and phase in {"experiment_loop", "final_report"}:
            return "PREFLIGHT_BLOCKED: big todo is required before entering execution/reporting phases"
        return None
    return (
        "PREFLIGHT_BLOCKED: big todo is not defined. First inspect minimal context, then call "
        "preflight(action='set', payload_json={\"goal\":\"...\",\"steps\":[{\"step\":\"...\",\"check\":\"...\"}]})"
    )


def decide_gate(
    *,
    target: str,
    phase: str,
    run_id: str | None,
    node_name: str | None,
    workspace: str | None,
) -> str | None:
    del phase
    if target in {"final_report", "end"} and not has_big_todo(run_id, node_name):
        return "PREFLIGHT_BLOCKED: big todo is required before final_report or end"
    if target == "end":
        ok, message = final_gate(run_id, node_name, workspace)
        if not ok:
            return message
    return None


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


def _set_big_todo(payload: dict[str, Any], run_id: str | None, node_name: str | None) -> str:
    goal = str(payload.get("goal") or "").strip()
    steps, error = _normalize_steps_payload(payload)
    if not goal:
        return "PREFLIGHT_ERROR: goal is required"
    if error:
        return error

    state = {
        "ready": True,
        "goal": goal,
        "steps": steps,
        "updated_at": _now(),
    }
    save_big_todo(run_id, node_name, state)
    return "PREFLIGHT_OK\n" + render_big_todo(state)


def _mark_step_done(payload: dict[str, Any], run_id: str | None, node_name: str | None) -> str:
    state = load_big_todo(run_id, node_name)
    if not state["ready"]:
        return "PREFLIGHT_ERROR: big todo is not defined"

    raw_step = payload.get("step")
    try:
        step_index = int(raw_step)
    except (TypeError, ValueError):
        return "PREFLIGHT_ERROR: step must be a 1-based integer"

    if step_index < 1 or step_index > len(state["steps"]):
        return f"PREFLIGHT_ERROR: step must be between 1 and {len(state['steps'])}"

    state["steps"][step_index - 1]["done"] = True
    state["updated_at"] = _now()
    save_big_todo(run_id, node_name, state)
    return "PREFLIGHT_OK\n" + render_big_todo(state)


def _normalize_state(payload: dict[str, Any]) -> dict[str, Any]:
    goal = str(payload.get("goal") or "").strip()
    steps, _error = _normalize_steps_payload(payload)
    return {
        "ready": bool(payload.get("ready") and goal and steps),
        "goal": goal,
        "steps": steps,
        "updated_at": str(payload.get("updated_at") or _now()),
    }


def _normalize_steps_payload(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    raw_steps = payload.get("steps")
    if raw_steps is None and "plan" in payload and "acceptance" in payload:
        raw_steps = _steps_from_legacy_payload(payload)

    if not isinstance(raw_steps, list):
        return [], "PREFLIGHT_ERROR: steps must be a JSON array"
    if not (MIN_STEPS <= len(raw_steps) <= MAX_STEPS):
        return [], f"PREFLIGHT_ERROR: steps must contain {MIN_STEPS}-{MAX_STEPS} items"

    steps: list[dict[str, Any]] = []
    for raw in raw_steps:
        if not isinstance(raw, dict):
            return [], "PREFLIGHT_ERROR: each step must be an object"
        step = str(raw.get("step") or raw.get("task") or "").strip()
        check = str(raw.get("check") or raw.get("pass") or "").strip()
        if not step:
            return [], "PREFLIGHT_ERROR: step text is required"
        if not check:
            return [], "PREFLIGHT_ERROR: step check is required"
        steps.append({"step": step, "check": check, "done": bool(raw.get("done"))})
    return steps, None


def _steps_from_legacy_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    plan = _string_list(payload.get("plan"), min_count=0, max_count=MAX_STEPS)
    acceptance = payload.get("acceptance") if isinstance(payload.get("acceptance"), list) else []
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(plan[:MAX_STEPS]):
        check = ""
        if index < len(acceptance):
            raw_check = acceptance[index]
            if isinstance(raw_check, dict):
                check = str(raw_check.get("text") or raw_check.get("content") or "").strip()
            else:
                check = str(raw_check).strip()
        steps.append({"step": item, "check": check or item, "done": False})
    return steps


def _empty_state() -> dict[str, Any]:
    return {"ready": False, "goal": "", "steps": [], "updated_at": _now()}


def _string_list(value: Any, min_count: int = MIN_STEPS, max_count: int = MAX_STEPS) -> list[str]:
    if not isinstance(value, list):
        return []
    output = [str(item).strip() for item in value if str(item).strip()]
    return output[:max_count] if len(output) >= min_count or min_count == 0 else output


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = [
    "big_todo_path",
    "decide_gate",
    "final_gate",
    "has_big_todo",
    "load_big_todo",
    "pre_tool_gate",
    "preflight_action",
    "record_tool_fact",
    "render_big_todo",
    "render_big_todo_for_prompt",
    "save_big_todo",
]
