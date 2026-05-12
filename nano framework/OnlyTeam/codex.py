"""OnlyTeam Codex-style single-agent entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STARTUP_CWD = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdin, "reconfigure") and not sys.stdin.isatty():
    sys.stdin.reconfigure(encoding="utf-8", errors="strict")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from framework import *  # noqa: F401,F403
from framework.big_todo import end_decide, todo
from framework.control_tools import chat
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
from framework.session import create_run_id, load_env_defaults, pick_run_id
from OnlyTeam.codex_prompt import codex_prompt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--no-session-prompt", action="store_true")
    return parser.parse_args()


load_env_defaults(ROOT)
args = _parse_args()

if args.run_id:
    selected_run_id = args.run_id
elif args.no_session_prompt:
    selected_run_id = create_run_id("onlyteam_codex")
else:
    selected_run_id = pick_run_id(
        title="Previous OnlyTeam Codex sessions",
        run_prefix="onlyteam_codex",
    ) or create_run_id("onlyteam_codex")

llm = deepseek_llm(
    model="deepseek-v4-pro",
    reasoning_effort="high",
    tool_choice="auto",
)

onlyteam_codex = node(
    name="onlyteam_codex",
    display_name="OnlyTeam Codex",
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
        chat,
        end_decide,
    ],
    prompt=bind_run_context_prompt(codex_prompt(), "onlyteam_codex", STARTUP_CWD),
    edges={"end": "Only use when the work is done or the user explicitly wants to stop."},
)

onlyteam_codex_graph = graph(
    nodes=[onlyteam_codex],
    run_id=selected_run_id,
    start_node="onlyteam_codex",
    state={"workspace": str(STARTUP_CWD)},
)

print("OnlyTeam Codex")
print("=" * 68)
print()
user_message = input("Please describe your request: ").strip()
print()
run(onlyteam_codex_graph, user_message=user_message, stream=args.stream)
