from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from DatasetTeam.acceptance_state import decide_gate, save_big_todo
from DatasetTeam.data_pipeline import (
    build_decision_records_file,
    export_openrlhf_preference_file,
    judge_decision_records_file,
    normalize_judge_payload,
    validate_openrlhf_export_file,
)
from DatasetTeam.team import build_dataset_team


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="datasetteam_test_"))
        self.workspace = self.temp_dir / "workspace"
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.mem_root = self.workspace / ".mem" / "onlyteam_fixture"
        self.mem_root.mkdir(parents=True, exist_ok=True)
        records = [
            {"role": "user", "content": "Please inspect the repo and fix the issue."},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path":"a.py","start_line":1,"end_line":10}',
                "thinking": "Need to inspect before editing.",
            },
            {
                "type": "token_usage",
                "response_id": "resp_1",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "# file: a.py\nprint('hello')",
            },
            {"role": "user", "content": "Now update the implementation carefully."},
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "apply_patch",
                "arguments": '{"patch":"*** Begin Patch\\n*** End Patch"}',
                "thinking": "Patch the file with the smallest safe change.",
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "patch applied",
            },
            {
                "type": "function_call",
                "call_id": "call_missing",
                "name": "read_file",
                "arguments": '{"path":"missing.py","start_line":1,"end_line":5}',
                "thinking": "This one will miss an output pair.",
            },
        ]
        (self.mem_root / "only_team.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        self.output_dir = self.workspace / "outputs"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_decision_records_pairs_calls_and_tracks_missing_outputs(self) -> None:
        manifest = build_decision_records_file(
            workspace=self.workspace,
            source_run_id="onlyteam_fixture",
            output_dir=self.output_dir,
        )

        decision_path = self.output_dir / "decision_records.jsonl"
        rows = [json.loads(line) for line in decision_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(manifest["decision_records_total"], 2)
        self.assertEqual(manifest["missing_outputs_skipped"], 1)
        self.assertEqual(rows[0]["source"]["response_id"], "resp_1")
        self.assertEqual(rows[0]["state"]["recent_user_message"], "Please inspect the repo and fix the issue.")
        self.assertEqual(rows[1]["state"]["recent_user_message"], "Now update the implementation carefully.")
        self.assertEqual(rows[1]["decision"]["tool_name"], "apply_patch")

    def test_normalize_judge_payload_rejects_invalid_acceptance(self) -> None:
        with self.assertRaises(ValueError):
            normalize_judge_payload(
                {
                    "score": 0.91,
                    "confidence": 0.8,
                    "reason": "looks good",
                    "failure_modes": [],
                    "chosen_response": "<action>x</action>",
                    "rejected_response": "",
                    "export_decision": {"openrlhf_preference": True},
                },
                canonical_chosen="<action>x</action>",
            )

    def test_judge_and_export_openrlhf_files(self) -> None:
        build_decision_records_file(
            workspace=self.workspace,
            source_run_id="onlyteam_fixture",
            output_dir=self.output_dir,
        )

        judge_manifest = judge_decision_records_file(
            decision_records_path=self.output_dir / "decision_records.jsonl",
            output_dir=self.output_dir,
            judge_fn=lambda record: {
                "score": 0.9 if record["decision"]["tool_name"] == "read_file" else 0.2,
                "confidence": 0.8,
                "reason": "good" if record["decision"]["tool_name"] == "read_file" else "bad",
                "failure_modes": [] if record["decision"]["tool_name"] == "read_file" else ["no_task_progress"],
                "chosen_response": "",
                "rejected_response": "<thinking>\nWrong plan\n</thinking>\n<action>\nnoop({})\n</action>",
                "export_decision": {"openrlhf_preference": True},
            },
        )
        export_manifest = export_openrlhf_preference_file(
            judge_records_path=self.output_dir / "judge_records.jsonl",
            output_dir=self.output_dir,
        )

        strict_rows = [
            json.loads(line)
            for line in (self.output_dir / "openrlhf_preference.strict.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rich_rows = [
            json.loads(line)
            for line in (self.output_dir / "openrlhf_preference.rich.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(judge_manifest["judged_total"], 2)
        self.assertEqual(judge_manifest["accepted_total"], 1)
        self.assertEqual(export_manifest["strict_samples"], 1)
        self.assertEqual(set(strict_rows[0].keys()), {"chosen", "rejected"})
        self.assertIn("User:", strict_rows[0]["chosen"])
        self.assertIn("Assistant:", strict_rows[0]["rejected"])
        self.assertEqual(rich_rows[0]["source_run_id"], "onlyteam_fixture")

        validated = validate_openrlhf_export_file(
            strict_path=self.output_dir / "openrlhf_preference.strict.jsonl",
            rich_path=self.output_dir / "openrlhf_preference.rich.jsonl",
            manifest_path=self.output_dir / "export_manifest.json",
        )
        self.assertEqual(validated["status"], "ok")

    def test_judge_errors_become_rejects(self) -> None:
        build_decision_records_file(
            workspace=self.workspace,
            source_run_id="onlyteam_fixture",
            output_dir=self.output_dir,
        )

        manifest = judge_decision_records_file(
            decision_records_path=self.output_dir / "decision_records.jsonl",
            output_dir=self.output_dir,
            judge_fn=lambda _record: {"oops": "missing required fields"},
        )
        rows = [
            json.loads(line)
            for line in (self.output_dir / "judge_records.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(manifest["judge_errors"], 2)
        self.assertEqual(rows[0]["judge"]["label"], "reject")
        self.assertFalse(rows[0]["export_decision"]["openrlhf_preference"])

    def test_phase_gate_requires_previous_artifacts(self) -> None:
        run_id = "dataset_run"
        node_name = "dataset_team"
        save_big_todo(
            run_id,
            node_name,
            {
                "ready": True,
                "goal": "export rl data",
                "steps": [{"step": "x", "check": "y", "done": True}],
            },
        )

        blocked = decide_gate(
            target="judge_preference",
            phase="trajectory_extract",
            run_id=run_id,
            node_name=node_name,
            workspace=str(self.workspace),
            output_dir=str(self.output_dir),
            source_run_id="onlyteam_fixture",
        )
        self.assertIn("decision_records.jsonl", blocked or "")

    def test_build_dataset_team_sets_default_state(self) -> None:
        graph = build_dataset_team(workspace=self.workspace, source_run_id="onlyteam_fixture")
        state = graph["state"]
        self.assertEqual(state["source_run_id"], "onlyteam_fixture")
        self.assertEqual(state["compat_mode"], "openrlhf_preference_strict")
        self.assertEqual(state["judge_model"], "deepseekv4-pro")

    def test_end_to_end_smoke_with_real_mem_run_if_available(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        real_files = sorted((repo_root / ".mem").glob("onlyteam_*\\only_team.json"))
        if not real_files:
            self.skipTest("no real OnlyTeam memory run available in repo")
        real_path = real_files[-1]
        source_run_id = real_path.parent.name
        output_dir = self.temp_dir / "real_smoke_output"

        build_decision_records_file(
            workspace=repo_root,
            source_run_id=source_run_id,
            output_dir=output_dir,
        )
        judge_decision_records_file(
            decision_records_path=output_dir / "decision_records.jsonl",
            output_dir=output_dir,
            judge_fn=lambda record: {
                "score": 0.85,
                "confidence": 0.8,
                "reason": "smoke accept",
                "failure_modes": [],
                "chosen_response": "",
                "rejected_response": "<thinking>\nslightly worse plan\n</thinking>\n<action>\nnoop({})\n</action>",
                "export_decision": {"openrlhf_preference": True},
            },
        )
        export_openrlhf_preference_file(
            judge_records_path=output_dir / "judge_records.jsonl",
            output_dir=output_dir,
        )

        self.assertTrue((output_dir / "decision_records.jsonl").exists())
        self.assertTrue((output_dir / "judge_records.jsonl").exists())
        self.assertTrue((output_dir / "openrlhf_preference.strict.jsonl").exists())
        self.assertTrue((output_dir / "export_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
