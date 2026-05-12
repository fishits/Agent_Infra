"""Boss entrypoint for dispatch generation."""

from __future__ import annotations

import argparse
import importlib.util
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

from framework import deepseek_llm, graph, node, run
from framework.big_todo import end_decide, todo
from framework.control_tools import chat
from framework.io_tools import glob_files, grep_search, list_directory, parallel_tools, read_file, read_many
from framework.prompts import bind_run_context_prompt
from framework.session import create_run_id, load_env_defaults, pick_run_id

from OnlyTeam.boss_tools import submit_parallel_dispatch


PROMPT_FILE = Path(__file__).with_name("boss.prompt.py")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--no-session-prompt", action="store_true")
    return parser.parse_args()


def _load_boss_prompt() -> str:
    spec = importlib.util.spec_from_file_location("OnlyTeam.boss_prompt_dynamic", PROMPT_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Boss prompt from {PROMPT_FILE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    prompt_fn = getattr(module, "boss_prompt", None)
    if not callable(prompt_fn):
        raise RuntimeError(f"{PROMPT_FILE} must define boss_prompt()")
    return str(prompt_fn())


load_env_defaults(ROOT)
args = _parse_args()


def build_boss_graph(run_id: str | None = None) -> dict:
    llm = deepseek_llm(
        model="deepseek-v4-pro",
        reasoning_effort="high",
        tool_choice="auto",
    )

    onlyteam_boss = node(
        name="onlyteam_boss",
        display_name="OnlyTeam Boss",
        llm=llm,
        tools=[
            read_file,
            read_many,
            list_directory,
            parallel_tools,
            glob_files,
            grep_search,
            todo,
            chat,
            submit_parallel_dispatch,
            end_decide,
        ],
        prompt=bind_run_context_prompt(_load_boss_prompt(), "onlyteam_boss", STARTUP_CWD),
        edges={"end": "Only use after dispatch JSON and all route task cards have been created."},
    )

    return graph(
        nodes=[onlyteam_boss],
        run_id=run_id or create_run_id("onlyteam_boss"),
        start_node="onlyteam_boss",
        state={"workspace": str(STARTUP_CWD), "cwd": str(STARTUP_CWD)},
    )


def main() -> None:
    if args.run_id:
        selected_run_id = args.run_id
    elif args.no_session_prompt:
        selected_run_id = create_run_id("onlyteam_boss")
    else:
        selected_run_id = pick_run_id(
            title="Previous OnlyTeam Boss sessions",
            run_prefix="onlyteam_boss",
        ) or create_run_id("onlyteam_boss")
    onlyteam_boss_graph = build_boss_graph(run_id=selected_run_id)

    print("OnlyTeam Boss")
    print("=" * 68)
    print()
    user_message = input("Please describe your dispatch request: ").strip()
    print()
    run(onlyteam_boss_graph, user_message=user_message, stream=args.stream)


if __name__ == "__main__":
    main()
