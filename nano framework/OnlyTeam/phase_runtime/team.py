"""Declarative OnlyTeam entrypoint.

Keep this file simple:
- choose one LLM in code
- declare one node with tools and prompt
- assemble the graph
- optionally run it from CLI
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_root_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_root_env()

from framework.deepseek_llm import deepseek_llm  # noqa: E402
from framework.io_tools import (  # noqa: E402
    append_file,
    apply_diff,
    apply_patch,
    check_background_powershell,
    glob_files,
    grep_search,
    list_directory,
    read_file,
    read_many,
    run_powershell,
    run_python,
    sleep_then_check_background_powershell,
    start_background_powershell,
    todo_read,
    todo_write,
    write_file,
)

from OnlyTeam.phase_runtime.graph import graph, node, run  # noqa: E402
from OnlyTeam.phase_runtime.phases import AGENT_NODE  # noqa: E402
from OnlyTeam.phase_runtime.prompt import only_team_prompt  # noqa: E402
from OnlyTeam.phase_runtime.tools import chat, decide, preflight  # noqa: E402


RUN_PREFIX = "onlyteam"
MEM_ROOT = ROOT / ".mem"


llm = deepseek_llm(
    model="deepseek-v4-pro",
    reasoning_effort="max",
    tool_choice="auto",
)


def only_team_tools():
    return [
        preflight,
        todo_read,
        todo_write,
        list_directory,
        glob_files,
        grep_search,
        read_file,
        read_many,
        run_python,
        run_powershell,
        start_background_powershell,
        check_background_powershell,
        sleep_then_check_background_powershell,
        write_file,
        append_file,
        apply_diff,
        apply_patch,
        chat,
        decide,
    ]


only_team = node(
    name=AGENT_NODE,
    display_name="OnlyTeam",
    llm=llm,
    tools=only_team_tools(),
    prompt=only_team_prompt(),
    edges={
        "end": "End only after final verification is complete.",
    },
)


def build_only_team(run_id: str | None = None, workspace: str | Path | None = None) -> dict:
    state: dict[str, str] = {}
    if workspace is not None:
        state["workspace"] = str(Path(workspace).resolve())
    return graph(
        nodes=[only_team],
        start_node=AGENT_NODE,
        run_id=run_id or _create_run_id(),
        state=state,
    )


def main() -> None:
    stream = "--stream" in sys.argv
    print("OnlyTeam")
    print("=" * 68)

    run_id = _pick_run_id()
    print()

    only_team_graph = build_only_team(run_id=run_id, workspace=ROOT)
    user_message = input("Please describe your request: ").strip()
    if not user_message:
        user_message = "Inspect this workspace and handle the next concrete task."
    print()
    run(only_team_graph, user_message=user_message, stream=stream)


def _create_run_id() -> str:
    return f"{RUN_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _format_run_label(run_id: str) -> str:
    stem = run_id[len(RUN_PREFIX) + 1 :] if run_id.startswith(f"{RUN_PREFIX}_") else run_id
    parts = stem.split("_")
    if len(parts) == 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
        day, clock = parts
        return f"{day[:4]}-{day[4:6]}-{day[6:8]} {clock[:2]}:{clock[2:4]}:{clock[4:6]}"
    return run_id


def _list_existing_runs() -> list[tuple[str, str]]:
    if not MEM_ROOT.is_dir():
        return []

    runs: list[tuple[str, str]] = []
    for child in sorted(MEM_ROOT.iterdir()):
        if not child.is_dir():
            continue
        history = child / f"{AGENT_NODE}.json"
        if not history.exists():
            continue
        try:
            records = json.loads(history.read_text(encoding="utf-8") or "[]")
        except json.JSONDecodeError:
            records = []
        record_count = len(records) if isinstance(records, list) else 0
        label = f"{_format_run_label(child.name)}  [{AGENT_NODE}; {record_count} records]"
        runs.append((child.name, label))
    return runs


def _pick_run_id() -> str | None:
    runs = _list_existing_runs()
    if not runs:
        print("No previous OnlyTeam sessions found. Starting a new session.")
        return None

    print("Previous OnlyTeam sessions:")
    print("-" * 60)
    for idx, (run_id, label) in enumerate(runs, 1):
        marker = " (latest)" if idx == len(runs) else ""
        print(f"  [{idx}] {label}{marker}")
    print("  [n] Start a new session")
    print("-" * 60)

    choice = input("Select session [Enter = latest]: ").strip().lower()
    if choice == "n":
        return None
    if choice == "":
        selected = runs[-1][0]
        print(f"  -> Resuming {selected}")
        return selected
    try:
        index = int(choice) - 1
    except ValueError:
        print("  -> Invalid choice, starting a new session.")
        return None
    if 0 <= index < len(runs):
        selected = runs[index][0]
        print(f"  -> Resuming {selected}")
        return selected
    print("  -> Invalid choice, starting a new session.")
    return None


if __name__ == "__main__":
    main()
