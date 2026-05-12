"""Declarative DatasetTeam entrypoint."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
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
from framework.io_tools import read_file, read_many, todo_read, todo_write  # noqa: E402
from framework.session import pick_run_id  # noqa: E402

from DatasetTeam.graph import graph, node, run  # noqa: E402
from DatasetTeam.phases import AGENT_NODE  # noqa: E402
from DatasetTeam.prompt import dataset_team_prompt  # noqa: E402
from DatasetTeam.tools import (  # noqa: E402
    build_decision_records,
    chat,
    decide,
    export_openrlhf_preference,
    judge_decision_records,
    list_dataset_runs,
    preflight,
    read_mem_run,
    validate_openrlhf_export,
)


RUN_PREFIX = "datasetteam"
MEM_ROOT = ROOT / ".mem"
OUTPUT_ROOT = ROOT / "outputs" / "dataset_team"


llm = deepseek_llm(
    model="deepseek-v4-pro",
    reasoning_effort="max",
    tool_choice="auto",
)


def dataset_team_tools():
    return [
        preflight,
        todo_read,
        todo_write,
        list_dataset_runs,
        read_mem_run,
        build_decision_records,
        judge_decision_records,
        export_openrlhf_preference,
        validate_openrlhf_export,
        read_file,
        read_many,
        chat,
        decide,
    ]


dataset_team = node(
    name=AGENT_NODE,
    display_name="DatasetTeam",
    llm=llm,
    tools=dataset_team_tools(),
    prompt=dataset_team_prompt(),
    edges={"end": "End only after strict export validation is complete."},
)


def build_dataset_team(
    run_id: str | None = None,
    workspace: str | Path | None = None,
    source_run_id: str | None = None,
    output_dir: str | Path | None = None,
    judge_model: str = "deepseekv4-pro",
) -> dict:
    resolved_workspace = Path(workspace or ROOT).resolve()
    resolved_run_id = run_id or _create_run_id()
    resolved_output_dir = Path(output_dir) if output_dir is not None else OUTPUT_ROOT / resolved_run_id
    state: dict[str, str] = {
        "workspace": str(resolved_workspace),
        "output_dir": str(resolved_output_dir.resolve()),
        "judge_model": judge_model,
        "judge_reasoning_effort": "medium",
        "compat_mode": "openrlhf_preference_strict",
    }
    if source_run_id is not None:
        state["source_run_id"] = source_run_id
    return graph(
        nodes=[dataset_team],
        start_node=AGENT_NODE,
        run_id=resolved_run_id,
        state=state,
    )


def main() -> None:
    print("DatasetTeam")
    print("=" * 68)
    run_id = pick_run_id(
        title="Previous DatasetTeam sessions",
        mem_root=MEM_ROOT,
        workspace_root=OUTPUT_ROOT if OUTPUT_ROOT.exists() else None,
        run_prefix=RUN_PREFIX,
    )
    source_run_id = input("Source OnlyTeam run id: ").strip()
    if not source_run_id:
        print("source_run_id is required")
        return
    dataset_team_graph = build_dataset_team(run_id=run_id, workspace=ROOT, source_run_id=source_run_id)
    user_message = input("Please describe your request: ").strip()
    if not user_message:
        user_message = f"Build RL preference exports from OnlyTeam run {source_run_id}."
    print()
    run(dataset_team_graph, user_message=user_message)


def _create_run_id() -> str:
    return f"{RUN_PREFIX}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


if __name__ == "__main__":
    main()
