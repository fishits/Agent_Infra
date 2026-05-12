"""OnlyTeam phase specifications.

The phase spec is the single source of truth for prompt mode cards,
runtime tool gates, and decide transitions.
"""

from __future__ import annotations


AGENT_NODE = "only_team"
START_PHASE = "task_setup"
END_PHASE = "end"


PHASE_SPECS: dict[str, dict[str, object]] = {
    "task_setup": {
        "display_name": "OnlyTeam Task Setup",
        "objective": "把用户请求收敛成最小可执行的数据科学任务定义。",
        "allowed_actions": [
            "回应问候、身份问题或明显不完整的非任务输入。",
            "轻量查看工作目录和输入文件，确认是否有足够信息开始画像。",
            "只在目标、指标或交付物确实无法判断时问一个关键问题。",
        ],
        "acceptance_gate": [
            "任务类型、目标或目标不确定性已经明确。",
            "默认指标、关键产物和硬约束已用不超过五项说明。",
            "有足够信息进入 data_profile，或已通过 chat 请求必要澄清。",
        ],
        "next": ["data_profile"],
        "allowed_tools": [
            "chat",
            "decide",
            "preflight",
            "todo_read",
            "todo_write",
            "list_directory",
            "glob_files",
            "grep_search",
            "read_file",
            "read_many",
            "run_python",
        ],
    },
    "data_profile": {
        "display_name": "OnlyTeam Data Profile",
        "objective": "建立数据事实，确认字段、目标、缺失、时间顺序和泄漏风险。",
        "allowed_actions": [
            "读取真实数据和轻量画像。",
            "识别输入文件、行列数、字段类型、目标候选、缺失和时间风险。",
            "在目标仍不明确时只问一个关键问题。",
        ],
        "acceptance_gate": [
            "至少一次真实数据检查成功。",
            "表格数据的行列数、目标缺失和关键字段风险已报告。",
            "时间/顺序/泄漏风险已明确适用或不适用。",
        ],
        "next": ["task_setup", "experiment_loop"],
        "allowed_tools": [
            "chat",
            "decide",
            "preflight",
            "todo_read",
            "todo_write",
            "list_directory",
            "glob_files",
            "grep_search",
            "read_file",
            "read_many",
            "run_python",
        ],
    },
    "experiment_loop": {
        "display_name": "OnlyTeam Experiment Loop",
        "objective": "写代码、运行、检查失败、修复复跑，并生成可验证产物。",
        "allowed_actions": [
            "创建或维护当前工作目录中的可运行入口，通常是 main.py。",
            "运行实验、检查 stderr/traceback、修复可修问题并复跑。",
            "在 outputs/ 或用户指定位置生成指标、表格、图、报告或模型产物。",
        ],
        "acceptance_gate": [
            "存在可运行入口或已证明本任务不需要入口。",
            "最近一次实质代码修改后至少运行过一次相关验证。",
            "成功必须由退出码和真实产物支持；失败必须至少修复复跑一次。",
        ],
        "next": ["data_profile", "final_report"],
        "allowed_tools": [
            "chat",
            "decide",
            "preflight",
            "todo_read",
            "todo_write",
            "list_directory",
            "glob_files",
            "grep_search",
            "read_file",
            "read_many",
            "run_python",
            "run_powershell",
            "start_background_powershell",
            "check_background_powershell",
            "sleep_then_check_background_powershell",
            "write_file",
            "append_file",
            "apply_diff",
            "apply_patch",
        ],
    },
    "final_report": {
        "display_name": "OnlyTeam Final Report",
        "objective": "检查最终产物并给出简洁、证据支持的结论。",
        "allowed_actions": [
            "核查入口、输出目录、指标、图表、报告和关键产物真实存在。",
            "总结验证过的结果、产物路径和剩余风险。",
            "如果验收缺失且可修，切回 experiment_loop。",
        ],
        "acceptance_gate": [
            "预期产物已经通过工具检查。",
            "最终答复说明入口、验证命令、结果和产物路径。",
            "只有最终核查完成后才允许切到 end。",
        ],
        "next": ["experiment_loop", "end"],
        "allowed_tools": [
            "chat",
            "decide",
            "preflight",
            "todo_read",
            "todo_write",
            "list_directory",
            "glob_files",
            "grep_search",
            "read_file",
            "read_many",
            "run_python",
            "write_file",
            "append_file",
            "apply_diff",
            "apply_patch",
        ],
    },
}


def phase_spec(phase: str) -> dict[str, object]:
    if phase not in PHASE_SPECS:
        raise ValueError(f"Unknown OnlyTeam phase: {phase}")
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
