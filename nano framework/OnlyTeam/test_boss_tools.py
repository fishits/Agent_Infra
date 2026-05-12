from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from OnlyTeam.boss_tools import _launch_subagent, submit_parallel_dispatch


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid


class BossDispatchTests(unittest.TestCase):
    def test_submit_parallel_dispatch_creates_summary_and_cards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_context = {
                "cwd": tmpdir,
                "workspace": tmpdir,
                "run_id": "onlyteam_boss_20260508_230000",
            }
            payload = {
                "task_title": "仅完成 N2O 的 30 分钟时序预测",
                "goal": "基于历史分钟级监控数据，预测未来 30 分钟的 N2O 数值。",
                "completion_criteria": ["已生成全部交付物", "结果支持可复核结论"],
                "information_sources": ["唯一数据源：HKU数据建模\\input.csv", "任务表格 JSON"],
                "constraints": ["仅完成 N2O", "CPU 最大核心数不超过 4"],
            }

            def fake_launch(*, task_card_path: Path, workspace: Path, weak_bias_label: str, dispatch_id: str, route_index: int) -> dict:
                return {
                    "status": "launched",
                    "run_id": f"{dispatch_id}_subagent_{route_index:02d}",
                    "pid": 1000 + route_index,
                    "log_path": str((workspace / "logs" / "subagent.log").resolve()),
                    "workspace": str(workspace),
                    "task_card_path": str(task_card_path),
                    "weak_bias": weak_bias_label,
                    "started_at": "2026-05-09 00:00:00",
                }

            with patch("OnlyTeam.boss_tools._launch_subagent", side_effect=fake_launch):
                result = submit_parallel_dispatch(
                    payload_json=json.dumps(payload, ensure_ascii=False),
                    runtime_context=runtime_context,
                )

            self.assertEqual(result["status"], "ok")
            self.assertTrue(runtime_context["parallel_ready"])
            self.assertEqual(len(runtime_context["route_workspaces"]), 5)
            self.assertEqual(len(runtime_context["route_task_files"]), 5)
            self.assertEqual(len(result["subagent_runs"]), 5)

            dispatch_path = Path(result["dispatch_path"])
            self.assertTrue(dispatch_path.exists())

            summary = json.loads(dispatch_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["task_title"], payload["task_title"])
            self.assertEqual(summary["weak_biases"], ["证据敏感", "风险敏感", "验收敏感", "探索增益敏感", "效率敏感"])
            self.assertEqual(len(summary["subagent_runs"]), 5)

            first_task_file = Path(result["route_task_files"][0])
            self.assertTrue(first_task_file.exists())
            card = json.loads(first_task_file.read_text(encoding="utf-8"))
            self.assertNotIn("dispatch_id", card)
            self.assertNotIn("agent_id", card)
            self.assertNotIn("recording_requirements", card)
            self.assertNotIn("report_requirements", card)
            self.assertIn("证据敏感", card["weak_bias"])
            self.assertEqual(card["workspace"], str(first_task_file.parent.resolve()))

    def test_submit_parallel_dispatch_rejects_agents_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_context = {"cwd": tmpdir, "run_id": "onlyteam_boss_20260508_230001"}
            payload = {
                "task_title": "任务",
                "goal": "目标",
                "completion_criteria": ["完成"],
                "information_sources": ["输入"],
                "constraints": ["约束"],
                "agents": ["证据敏感"],
            }

            with self.assertRaisesRegex(ValueError, "unexpected keys: agents"):
                submit_parallel_dispatch(
                    payload_json=json.dumps(payload, ensure_ascii=False),
                    runtime_context=runtime_context,
                )

    def test_submit_parallel_dispatch_uses_fixed_five_biases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_context = {"cwd": tmpdir, "run_id": "onlyteam_boss_20260508_230002"}
            payload = {
                "task_title": "任务",
                "goal": "目标",
                "completion_criteria": ["完成"],
                "information_sources": ["输入"],
                "constraints": ["约束"],
            }

            def fake_launch(*, task_card_path: Path, workspace: Path, weak_bias_label: str, dispatch_id: str, route_index: int) -> dict:
                return {
                    "status": "launched",
                    "run_id": f"{dispatch_id}_subagent_{route_index:02d}",
                    "pid": 2000 + route_index,
                    "log_path": str((workspace / "logs" / "subagent.log").resolve()),
                    "workspace": str(workspace),
                    "task_card_path": str(task_card_path),
                    "weak_bias": weak_bias_label,
                    "started_at": "2026-05-09 00:00:00",
                }

            with patch("OnlyTeam.boss_tools._launch_subagent", side_effect=fake_launch):
                result = submit_parallel_dispatch(
                    payload_json=json.dumps(payload, ensure_ascii=False),
                    runtime_context=runtime_context,
                )

            self.assertEqual(
                result["route_biases"],
                ["证据敏感", "风险敏感", "验收敏感", "探索增益敏感", "效率敏感"],
            )
            self.assertEqual(len(result["route_workspaces"]), 5)

    def test_launch_subagent_writes_launch_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir).resolve()
            (workspace / "logs").mkdir(parents=True, exist_ok=True)
            task_card_path = workspace / "task_card.json"
            task_card_path.write_text("{}", encoding="utf-8")

            def fake_start_logged_process(*, command, log_path, command_text, env=None, cwd=None):
                del command_text, env, cwd
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text("[launcher] started\n", encoding="utf-8")
                self.assertIn("--task-card", command)
                self.assertIn("--workspace", command)
                self.assertIn("--run-id", command)
                return _FakeProcess(pid=4321)

            with patch("OnlyTeam.boss_tools._start_logged_process", side_effect=fake_start_logged_process):
                launch_info = _launch_subagent(
                    task_card_path=task_card_path,
                    workspace=workspace,
                    weak_bias_label="证据敏感",
                    dispatch_id="onlyteam_boss_20260509_123000",
                    route_index=1,
                )

            launch_file = workspace / "subagent_launch.json"
            self.assertTrue(launch_file.exists())
            self.assertEqual(launch_info["pid"], 4321)
            self.assertEqual(launch_info["workspace"], str(workspace))
            self.assertEqual(launch_info["task_card_path"], str(task_card_path))
            self.assertEqual(launch_info["weak_bias"], "证据敏感")
            self.assertEqual(
                json.loads(launch_file.read_text(encoding="utf-8"))["run_id"],
                "onlyteam_boss_20260509_123000_subagent_01",
            )


if __name__ == "__main__":
    unittest.main()
