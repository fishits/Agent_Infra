"""DatasetTeam control and data tools."""

from __future__ import annotations

import json
from typing import Any

from framework.tools import tool

from DatasetTeam.acceptance_state import preflight_action
from DatasetTeam.data_pipeline import (
    build_decision_records_file,
    export_openrlhf_preference_file,
    judge_decision_records_file,
    list_dataset_runs as list_dataset_runs_impl,
    read_mem_run as read_mem_run_impl,
    validate_openrlhf_export_file,
)
from DatasetTeam.phases import END_PHASE, known_phase_names, next_phases


PHASE_TARGETS = known_phase_names()


@tool(
    description=(
        "Switch the DatasetTeam work mode. This keeps the same shared conversation history; "
        "it only updates state['phase']."
    ),
    params={
        "target": (
            "string",
            "Next phase: task_setup, trajectory_extract, judge_preference, export_rl, final_report, or end",
        ),
        "reason": ("string", "Short reason shown in the CLI route display."),
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
        return "DECIDE_ERROR: runtime_context.phase is required for DatasetTeam mode switch"
    if phase == END_PHASE:
        return "DECIDE_ERROR: current phase is already end"
    allowed_targets = set(next_phases(phase))
    if target not in allowed_targets:
        allowed = ", ".join(sorted(allowed_targets))
        return f"DECIDE_ERROR: current phase {phase} can only route to: {allowed}"
    return f"DECIDE:{target}:{reason or ''}"


@tool(
    description="Ask the user for one concise clarification and pause the current phase.",
    params={"message": ("string", "Message shown to the user.")},
)
def chat(message: str) -> str:
    return f"CHAT:{message}"


@tool(
    description=(
        "Manage the DatasetTeam big todo before and during work. "
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


@tool(
    description="List available OnlyTeam source runs under .mem that can be converted into RL preference data.",
    params={},
)
def list_dataset_runs(runtime_context: dict[str, Any] | None = None) -> str:
    context = runtime_context or {}
    payload = list_dataset_runs_impl(context.get("workspace"))
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool(
    description="Read one source OnlyTeam memory run and summarize record counts and path info.",
    params={"source_run_id": ("string", "Existing OnlyTeam run id under .mem/")},
)
def read_mem_run(source_run_id: str, runtime_context: dict[str, Any] | None = None) -> str:
    context = runtime_context or {}
    payload = read_mem_run_impl(context.get("workspace"), source_run_id)
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool(
    description="Build DecisionRecord.jsonl from one OnlyTeam source run.",
    params={"source_run_id": ("string", "Existing OnlyTeam run id under .mem/")},
)
def build_decision_records(source_run_id: str, runtime_context: dict[str, Any] | None = None) -> str:
    context = runtime_context or {}
    payload = build_decision_records_file(
        workspace=context.get("workspace"),
        source_run_id=source_run_id,
        output_dir=context.get("output_dir"),
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool(
    description="Judge decision_records.jsonl and produce judge_records.jsonl for RL preference export.",
    params={
        "decision_records_path": ("string", "Path to decision_records.jsonl"),
        "judge_model": ("string", "OpenAI judge model id"),
        "judge_reasoning_effort": ("string", "Judge reasoning effort"),
    },
)
def judge_decision_records(
    decision_records_path: str,
    judge_model: str,
    judge_reasoning_effort: str,
    runtime_context: dict[str, Any] | None = None,
) -> str:
    context = runtime_context or {}
    payload = judge_decision_records_file(
        decision_records_path=decision_records_path,
        output_dir=context.get("output_dir"),
        judge_model=judge_model,
        judge_reasoning_effort=judge_reasoning_effort,
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool(
    description="Export strict and rich OpenRLHF preference JSONL files from judge_records.jsonl.",
    params={"judge_records_path": ("string", "Path to judge_records.jsonl")},
)
def export_openrlhf_preference(judge_records_path: str, runtime_context: dict[str, Any] | None = None) -> str:
    context = runtime_context or {}
    payload = export_openrlhf_preference_file(
        judge_records_path=judge_records_path,
        output_dir=context.get("output_dir"),
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool(
    description="Validate the strict OpenRLHF preference export and compare it with the manifest.",
    params={
        "strict_path": ("string", "Path to openrlhf_preference.strict.jsonl"),
        "rich_path": ("string", "Path to openrlhf_preference.rich.jsonl"),
        "manifest_path": ("string", "Path to export_manifest.json"),
    },
)
def validate_openrlhf_export(
    strict_path: str,
    rich_path: str,
    manifest_path: str,
    runtime_context: dict[str, Any] | None = None,
) -> str:
    del runtime_context
    payload = validate_openrlhf_export_file(
        strict_path=strict_path,
        rich_path=rich_path,
        manifest_path=manifest_path,
    )
    return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "PHASE_TARGETS",
    "build_decision_records",
    "chat",
    "decide",
    "export_openrlhf_preference",
    "judge_decision_records",
    "list_dataset_runs",
    "preflight",
    "read_mem_run",
    "validate_openrlhf_export",
]
