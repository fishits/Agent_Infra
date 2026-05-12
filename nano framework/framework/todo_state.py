"""Persistent todo state for one agent session."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .mem_paths import MEM_ROOT, run_dir

VALID_STATUSES = {"pending", "in_progress", "completed"}


def normalize_todos(payload: Any) -> tuple[list[dict[str, str]] | None, str | None]:
    """Validate and normalize a TodoWrite-style JSON payload."""
    if not isinstance(payload, list):
        return None, "todos_json must be a JSON array"

    normalized: list[dict[str, str]] = []
    in_progress_count = 0
    for item in payload:
        if not isinstance(item, dict):
            return None, "each todo must be an object"
        content = str(item.get("content") or "").strip()
        status = str(item.get("status") or "").strip()
        active_form = str(item.get("activeForm") or "").strip()
        if not content:
            return None, "todo content is required"
        if status not in VALID_STATUSES:
            return None, f"invalid todo status: {status}"
        if not active_form:
            return None, "todo activeForm is required"
        if status == "in_progress":
            in_progress_count += 1
        normalized.append({"content": content, "status": status, "activeForm": active_form})

    if normalized and in_progress_count != 1:
        return None, "todo list must have exactly one in_progress item"
    return normalized, None


def read_todo_state(run_id: str | None, node_name: str | None) -> list[dict[str, str]]:
    if not run_id or not node_name:
        return []
    path = _todo_path(run_id, node_name)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    todos, error = normalize_todos(payload)
    if error or todos is None:
        return []
    return todos


def write_todo_state(run_id: str | None, node_name: str | None, todos: list[dict[str, str]]) -> None:
    if not run_id or not node_name:
        return
    path = _todo_path(run_id, node_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(todos, ensure_ascii=False, indent=2), encoding="utf-8")


def render_todo_state_for_context(run_id: str, node_name: str) -> dict | None:
    todos = read_todo_state(run_id, node_name)
    if not todos:
        return None
    lines = ["Current plan:"]
    for item in todos:
        lines.append(f"- [{item['status']}] {item['content']}")
    return {"role": "user", "content": "\n".join(lines)}


def _todo_path(run_id: str, node_name: str) -> Path:
    safe = node_name.replace("/", "_").replace("\\", "_")
    return run_dir(run_id, MEM_ROOT) / f"{safe}.todo.json"


__all__ = [
    "normalize_todos",
    "read_todo_state",
    "render_todo_state_for_context",
    "write_todo_state",
]
