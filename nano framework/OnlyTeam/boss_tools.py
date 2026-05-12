"""Boss dispatch tools for multi-agent route preparation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

from framework.io_tools import _now_text, _start_logged_process
from framework.tools import tool


WEAK_BIAS_LIBRARY = {
    "证据敏感": "证据敏感：当存在多种可行路径且当前证据不足以唯一决定下一步时，更优先关注哪些判断已被证据支持或证伪。这不是方法要求；如果后续证据表明其他方向更合理，你可以立即调整。",
    "风险敏感": "风险敏感：当存在多种可行路径且当前证据不足以唯一决定下一步时，更优先警惕高代价错误、伪成功和难回滚决策。这不是方法要求；如果后续证据表明其他方向更合理，你可以立即调整。",
    "验收敏感": "验收敏感：当存在多种可行路径且当前证据不足以唯一决定下一步时，更优先关注哪些因素最影响最终是否算完成。这不是方法要求；如果后续证据表明其他方向更合理，你可以立即调整。",
    "探索增益敏感": "探索增益敏感：当存在多种可行路径且当前证据不足以唯一决定下一步时，更优先选择最能增加信息量或区分假设的动作。这不是方法要求；如果后续证据表明其他方向更合理，你可以立即调整。",
    "效率敏感": "效率敏感：当存在多种可行路径且当前证据不足以唯一决定下一步时，更优先选择验证成本更低、推进速度更快的动作。这不是方法要求；如果后续证据表明其他方向更合理，你可以立即调整。",
}
DEFAULT_WEAK_BIASES = list(WEAK_BIAS_LIBRARY.keys())
REPO_ROOT = Path(__file__).resolve().parent.parent
SUBAGENT_ENTRY = Path(__file__).with_name("subagent.py")


@tool(
    description=(
        "提交 Boss 的并行分发结果。"
        "输入共享任务字段和 5 个 agent 的弱偏置标签，工具会自动生成 5 个独立 workspace、"
        "5 份 flat task card JSON，并把 dispatch 路径写回 graph state。"
    ),
    params={
        "payload_json": (
            "string",
            "JSON object with task_title, goal, completion_criteria, information_sources, constraints."
            " The tool always fans out to 5 fixed weak_bias routes: "
            "证据敏感, 风险敏感, 验收敏感, 探索增益敏感, 效率敏感.",
        ),
    },
)
def submit_parallel_dispatch(
    payload_json: str,
    runtime_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = runtime_context or {}
    try:
        payload = json.loads(payload_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"payload_json must be valid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError("payload_json must be a JSON object")

    task_title = _require_string(payload, "task_title")
    goal = _require_string(payload, "goal")
    completion_criteria = _require_string_list(payload, "completion_criteria", min_count=1)
    information_sources = _require_string_list(payload, "information_sources", min_count=1)
    constraints = _require_string_list(payload, "constraints", min_count=1)
    _reject_unexpected_keys(payload, allowed_keys={"task_title", "goal", "completion_criteria", "information_sources", "constraints"})

    cwd = _workspace_root(context)
    dispatch_id = str(context.get("run_id") or "boss_dispatch").strip() or "boss_dispatch"
    dispatch_root = (cwd / "coding" / "dispatches" / dispatch_id).resolve()
    dispatch_root.mkdir(parents=True, exist_ok=True)

    task_files: list[str] = []
    workspaces: list[str] = []
    weak_biases: list[str] = []
    subagent_runs: list[dict[str, Any]] = []

    for index, weak_bias_label in enumerate(DEFAULT_WEAK_BIASES, start=1):
        route_dir = dispatch_root / f"{index:02d}_agent"
        for name in ("outputs", "artifacts", "logs"):
            (route_dir / name).mkdir(parents=True, exist_ok=True)

        task_file = route_dir / "task_card.json"
        task_card = {
            "task_title": task_title,
            "goal": goal,
            "completion_criteria": completion_criteria,
            "information_sources": information_sources,
            "constraints": constraints,
            "weak_bias": WEAK_BIAS_LIBRARY[weak_bias_label],
            "workspace": str(route_dir.resolve()),
        }
        task_file.write_text(json.dumps(task_card, ensure_ascii=False, indent=2), encoding="utf-8")

        task_files.append(str(task_file.resolve()))
        workspaces.append(str(route_dir.resolve()))
        weak_biases.append(weak_bias_label)
        subagent_runs.append(
            _launch_subagent(
                task_card_path=task_file.resolve(),
                workspace=route_dir.resolve(),
                weak_bias_label=weak_bias_label,
                dispatch_id=dispatch_id,
                route_index=index,
            )
        )

    dispatch_summary = {
        "dispatch_id": dispatch_id,
        "task_title": task_title,
        "goal": goal,
        "completion_criteria": completion_criteria,
        "information_sources": information_sources,
        "constraints": constraints,
        "weak_biases": weak_biases,
        "workspaces": workspaces,
        "task_files": task_files,
        "subagent_runs": subagent_runs,
    }
    dispatch_path = dispatch_root / "boss_dispatch.json"
    dispatch_path.write_text(json.dumps(dispatch_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    context["dispatch_path"] = str(dispatch_path.resolve())
    context["route_workspaces"] = workspaces
    context["route_task_files"] = task_files
    context["route_biases"] = weak_biases
    context["subagent_runs"] = subagent_runs
    context["parallel_ready"] = True

    return {
        "status": "ok",
        "dispatch_id": dispatch_id,
        "dispatch_path": str(dispatch_path.resolve()),
        "route_workspaces": workspaces,
        "route_task_files": task_files,
        "route_biases": weak_biases,
        "subagent_runs": subagent_runs,
    }


def _launch_subagent(
    *,
    task_card_path: Path,
    workspace: Path,
    weak_bias_label: str,
    dispatch_id: str,
    route_index: int,
) -> dict[str, Any]:
    run_id = f"{dispatch_id}_subagent_{route_index:02d}"
    log_path = (workspace / "logs" / "subagent.log").resolve()
    command = [
        sys.executable,
        "-u",
        str(SUBAGENT_ENTRY.resolve()),
        "--task-card",
        str(task_card_path),
        "--workspace",
        str(workspace),
        "--run-id",
        run_id,
    ]
    command_text = " ".join(_quote_cmd_part(part) for part in command)
    process = _start_logged_process(
        command=command,
        log_path=log_path,
        command_text=command_text,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
        cwd=workspace,
    )
    launch_info = {
        "status": "launched",
        "run_id": run_id,
        "pid": process.pid,
        "log_path": str(log_path),
        "workspace": str(workspace),
        "task_card_path": str(task_card_path),
        "weak_bias": weak_bias_label,
        "started_at": _now_text(),
    }
    (workspace / "subagent_launch.json").write_text(
        json.dumps(launch_info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return launch_info


def _workspace_root(runtime_context: dict[str, Any]) -> Path:
    raw = runtime_context.get("cwd") or runtime_context.get("workspace")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).resolve()
    return Path.cwd().resolve()


def _require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _require_string_list(payload: dict[str, Any], key: str, min_count: int) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a JSON array")
    output = [str(item).strip() for item in value if isinstance(item, str) and str(item).strip()]
    if len(output) < min_count:
        raise ValueError(f"{key} must contain at least {min_count} non-empty strings")
    return output


def _reject_unexpected_keys(payload: dict[str, Any], allowed_keys: set[str]) -> None:
    extras = sorted(key for key in payload if key not in allowed_keys)
    if extras:
        raise ValueError(f"unexpected keys: {', '.join(extras)}")


def _quote_cmd_part(value: str) -> str:
    text = str(value)
    if not text or any(char.isspace() for char in text) or '"' in text:
        return '"' + text.replace('"', '\\"') + '"'
    return text


__all__ = [
    "DEFAULT_WEAK_BIASES",
    "WEAK_BIAS_LIBRARY",
    "submit_parallel_dispatch",
]
