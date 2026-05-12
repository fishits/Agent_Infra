"""DatasetTeam phase specifications."""

from __future__ import annotations


AGENT_NODE = "dataset_team"
START_PHASE = "task_setup"
END_PHASE = "end"


PHASE_SPECS: dict[str, dict[str, object]] = {
    "task_setup": {
        "display_name": "DatasetTeam Task Setup",
        "objective": "Confirm the source OnlyTeam run, output directory, judge configuration, and big todo before extraction.",
        "allowed_actions": [
            "Inspect available OnlyTeam runs and read the target memory file.",
            "Summarize the RL preference export goal into a compact big todo.",
            "Ask for one critical clarification only if source_run_id or judge intent is truly missing.",
        ],
        "acceptance_gate": [
            "source_run_id is known and points to an existing .mem/<run_id>/only_team.json file.",
            "output_dir and judge settings are visible in runtime context.",
            "big todo is defined and ready to enter trajectory_extract.",
        ],
        "next": ["trajectory_extract"],
        "allowed_tools": [
            "chat",
            "decide",
            "preflight",
            "todo_read",
            "todo_write",
            "list_dataset_runs",
            "read_mem_run",
            "read_file",
            "read_many",
        ],
    },
    "trajectory_extract": {
        "display_name": "DatasetTeam Trajectory Extract",
        "objective": "Build DecisionRecord.jsonl from one OnlyTeam memory file by pairing tool calls with outputs.",
        "allowed_actions": [
            "Read the target memory file and inspect record structure.",
            "Build decision_records.jsonl and report extraction counts.",
            "Route back only if source data is missing or malformed.",
        ],
        "acceptance_gate": [
            "decision_records.jsonl exists in output_dir.",
            "At least one paired tool-call record was extracted.",
            "Skipped items and missing outputs are summarized.",
        ],
        "next": ["task_setup", "judge_preference"],
        "allowed_tools": [
            "chat",
            "decide",
            "preflight",
            "todo_read",
            "todo_write",
            "read_mem_run",
            "build_decision_records",
            "read_file",
            "read_many",
        ],
    },
    "judge_preference": {
        "display_name": "DatasetTeam Judge Preference",
        "objective": "Use a frontier judge to score extracted decision records and construct rejected responses.",
        "allowed_actions": [
            "Read decision_records.jsonl and judge each record.",
            "Reject malformed judge output instead of exporting it.",
            "Route back only if extraction output is missing or invalid.",
        ],
        "acceptance_gate": [
            "judge_records.jsonl exists in output_dir.",
            "Judge results include score, confidence, label, reason, failure_modes, and rejected_response.",
            "Judge failures are counted separately from accepted/exportable items.",
        ],
        "next": ["trajectory_extract", "export_rl"],
        "allowed_tools": [
            "chat",
            "decide",
            "preflight",
            "todo_read",
            "todo_write",
            "judge_decision_records",
            "read_file",
            "read_many",
        ],
    },
    "export_rl": {
        "display_name": "DatasetTeam Export RL",
        "objective": "Export strict and rich OpenRLHF preference datasets plus a manifest.",
        "allowed_actions": [
            "Read judge_records.jsonl and filter exportable preference samples.",
            "Write strict and rich JSONL outputs and an export manifest.",
            "Route back only if judge output is missing or empty.",
        ],
        "acceptance_gate": [
            "openrlhf_preference.strict.jsonl exists and contains non-empty chosen/rejected fields only.",
            "openrlhf_preference.rich.jsonl and export_manifest.json exist.",
            "Exported sample counts match manifest counts.",
        ],
        "next": ["judge_preference", "final_report"],
        "allowed_tools": [
            "chat",
            "decide",
            "preflight",
            "todo_read",
            "todo_write",
            "export_openrlhf_preference",
            "read_file",
            "read_many",
        ],
    },
    "final_report": {
        "display_name": "DatasetTeam Final Report",
        "objective": "Validate export compatibility and report the final artifacts and training handoff.",
        "allowed_actions": [
            "Validate the strict OpenRLHF export line by line.",
            "Read the manifest and summarize artifact paths and counts.",
            "If validation fails, route back to export_rl.",
        ],
        "acceptance_gate": [
            "validate_openrlhf_export passes.",
            "The final reply names the strict export file, manifest, and sample counts.",
            "Only after validation may the run route to end.",
        ],
        "next": ["export_rl", "end"],
        "allowed_tools": [
            "chat",
            "decide",
            "preflight",
            "todo_read",
            "todo_write",
            "validate_openrlhf_export",
            "read_file",
            "read_many",
        ],
    },
}


def phase_spec(phase: str) -> dict[str, object]:
    if phase not in PHASE_SPECS:
        raise ValueError(f"Unknown DatasetTeam phase: {phase}")
    return PHASE_SPECS[phase]


def allowed_tool_names(phase: str) -> set[str]:
    return {str(name) for name in phase_spec(phase)["allowed_tools"]}


def next_phases(phase: str) -> list[str]:
    return [str(name) for name in phase_spec(phase)["next"]]


def phase_display_name(phase: str) -> str:
    if phase == END_PHASE:
        return "End"
    return str(phase_spec(phase)["display_name"])


def known_phase_names() -> set[str]:
    return {*PHASE_SPECS.keys(), END_PHASE}


__all__ = [
    "AGENT_NODE",
    "END_PHASE",
    "PHASE_SPECS",
    "START_PHASE",
    "allowed_tool_names",
    "known_phase_names",
    "next_phases",
    "phase_display_name",
    "phase_spec",
]
