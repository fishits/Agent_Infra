"""Fixed-background OnlyTeam sub-agent entrypoint."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from framework import deepseek_llm, graph, node, run
from framework.big_todo import end_decide, todo
from framework.io_tools import (
    append_file,
    apply_diff,
    apply_patch,
    check_background_powershell,
    glob_files,
    grep_search,
    list_directory,
    parallel_tools,
    read_file,
    read_many,
    run_powershell,
    run_python,
    sleep_then_check_background_powershell,
    start_background_powershell,
    write_file,
)
from framework.prompts import bind_run_context_prompt
from framework.session import load_env_defaults


PROMPT_FILE = Path(__file__).with_name("subagent.prompt.py")


def _load_subagent_prompt() -> str:
    spec = importlib.util.spec_from_file_location("OnlyTeam.subagent_prompt_dynamic", PROMPT_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load sub-agent prompt from {PROMPT_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt_fn = getattr(module, "subagent_prompt", None)
    if not callable(prompt_fn):
        raise RuntimeError(f"{PROMPT_FILE} must define subagent_prompt()")
    return str(prompt_fn())


def _build_graph(*, run_id: str, workspace: Path, task_card_path: Path, task_card: dict) -> dict:
    llm = deepseek_llm(
        model="deepseek-v4-pro",
        reasoning_effort="high",
        tool_choice="auto",
    )
    onlyteam_subagent = node(
        name="onlyteam_subagent",
        display_name="OnlyTeam SubAgent",
        llm=llm,
        tools=[
            read_file,
            read_many,
            list_directory,
            parallel_tools,
            glob_files,
            grep_search,
            write_file,
            append_file,
            apply_diff,
            apply_patch,
            run_python,
            run_powershell,
            start_background_powershell,
            check_background_powershell,
            sleep_then_check_background_powershell,
            todo,
            end_decide,
        ],
        prompt=bind_run_context_prompt(_load_subagent_prompt(), "onlyteam_subagent", workspace),
        edges={"end": "Only use when the assigned task card is truly complete."},
    )
    return graph(
        nodes=[onlyteam_subagent],
        run_id=run_id,
        start_node="onlyteam_subagent",
        state={
            "workspace": str(workspace),
            "cwd": str(workspace),
            "task_card_path": str(task_card_path),
            "weak_bias": str(task_card.get("weak_bias") or ""),
        },
    )


def _build_initial_message(task_card_path: Path, task_card: dict) -> str:
    task_json = json.dumps(task_card, ensure_ascii=False, indent=2)
    return (
        "你现在是固定执行 Sub-Agent。请只在当前 workspace 内完成这张任务卡。\n\n"
        f"task_card_path: {task_card_path}\n\n"
        "任务卡如下：\n"
        f"{task_json}\n\n"
        "先读取任务卡和当前 workspace 现状，再开始执行。复杂任务先建立 todo。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one fixed OnlyTeam sub-agent from a task card.")
    parser.add_argument("--task-card", required=True, dest="task_card")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", required=True, dest="run_id")
    args = parser.parse_args()

    load_env_defaults(ROOT)

    task_card_path = Path(args.task_card).resolve()
    workspace = Path(args.workspace).resolve()
    if not task_card_path.exists():
        raise FileNotFoundError(f"task card not found: {task_card_path}")
    if not workspace.exists():
        raise FileNotFoundError(f"workspace not found: {workspace}")

    task_card = json.loads(task_card_path.read_text(encoding="utf-8"))
    if not isinstance(task_card, dict):
        raise ValueError("task card must be a JSON object")

    subagent_graph = _build_graph(
        run_id=str(args.run_id),
        workspace=workspace,
        task_card_path=task_card_path,
        task_card=task_card,
    )
    run(
        subagent_graph,
        user_message=_build_initial_message(task_card_path, task_card),
        stream=True,
    )


if __name__ == "__main__":
    main()
