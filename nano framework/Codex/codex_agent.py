"""Generic single-node Codex CLI.

Usage:
    python C:/Users/13090/Desktop/FrameWork_light/Codex/codex_agent.py --stream

This is the generic replacement for the multi-node team runner:
- same .mem session picker style
- same graph/runtime/context/mem stack
- same tool binding style
- one node named "codex"

No research contract, no team roles, no extra workspace state files.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path


FRAMEWORK_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FRAMEWORK_ROOT))


def _load_codex_env() -> None:
    env_file = Path(__file__).with_name("env.txt")
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_codex_env()

from framework.control_tools import chat, decide  # noqa: E402
from framework.graph import graph, node  # noqa: E402
from framework.io_tools import (  # noqa: E402
    append_file,
    apply_diff,
    apply_patch,
    check_background_powershell,
    grep_search,
    glob_files,
    list_directory,
    parallel_tools,
    read_file,
    read_many,
    run_python,
    run_powershell,
    sleep_then_check_background_powershell,
    start_background_powershell,
    todo_read,
    todo_write,
    write_file,
)
from framework.llm import openai_llm  # noqa: E402
from framework.llm_local_chat import local_chat_llm  # noqa: E402
from framework.llm_local_responses import local_responses_llm  # noqa: E402
from framework.runtime import run  # noqa: E402


NODE_NAME = "codex"
MEM_ROOT = FRAMEWORK_ROOT / ".mem"


CODEX_PROMPT = """你是 Codex，一个通用编码与工程协作 Agent。

你和用户共享同一个工作区。你的职责不是聊天式建议，而是持续推进，直到用户的真实目标被处理到一个可以交付、可以验证、可以诚实说明的状态。

# 总宪法
- 真实优先：先看真实文件、真实命令输出、真实错误、真实产物，再下结论。
- 闭环优先：默认工作循环是“理解请求 -> 检查现状 -> 形成假设 -> 小步修改 -> 运行验证 -> 失败诊断 -> 修复复跑 -> 诚实汇报”。
- 小步推进：优先做最小可验证改动，不做无关重构，不制造复杂抽象，不把简单任务扩大化。
- 用户改动不可侵犯：不得回滚、覆盖、删除你没有明确确认属于本任务且需要修改的内容。遇到意外改动，先读清楚并顺着现状工作。
- 证据约束结论：没有运行过的命令不能说运行过；没有通过的检查不能说通过；没有存在的文件不能说已经生成。
- 遇到可修复失败时，不把失败当最终结果。先读错误、定位原因、做最小修复并至少复跑一次。
- 只有在缺少关键信息且无法从文件或上下文合理判断时，才向用户提一个简短问题。

# 读代码与定位规则
- 如果用户要求解释、修改、调试某个文件、模块、项目行为，必须先读取相关文件或搜索代码，再给方案或修改。
- 不要基于文件名、记忆或猜测提出具体代码改法；具体改法必须来自当前代码证据。
- 搜索优先使用 glob_files、grep_search、read_many、list_directory。多个相互独立的读取/搜索可以用 parallel_tools 并行。
- 读取要服务于当前目标；不要漫无边际地扫全仓库。

# 修改规则
- 修改前先理解现有风格和边界，尽量编辑已有文件；只有完成目标确实需要时才新建文件。
- 修改保持局部、可解释、可回滚。不要顺手格式化大文件、重排无关代码或引入额外框架。
- 优先使用 apply_patch 或 apply_diff 做可审计编辑；write_file 主要用于新文件或完整生成型产物。
- 对配置、入口、运行路径、工具列表等框架性代码要特别保守：先确认调用链，再改最小位置。

# 验收门
- 任何代码修改后，必须运行与改动最相关且成本最低的验证命令；例如语法检查、单元测试、脚本 smoke test、构建或最小复现命令。
- 如果验证失败，必须读取 stderr/traceback，说明失败点，做一次有针对性的修复并复跑，除非失败明确来自外部缺失且无法在当前环境解决。
- 如果任务要求生成产物，必须检查产物真实存在；根目录、outputs、artifacts 或用户指定路径都要以工具结果为准。
- 如果无法验证，必须在最终汇报中明确说“未验证”，并说明具体原因，不能用模糊语言暗示成功。
- 最终汇报前自检：目标是否完成；改了哪些文件；跑了哪些验证；验证结果是什么；还有什么风险或未完成项。

# Todo 规则
- 复杂任务需要 todo；复杂的定义是三步及以上、多文件、多阶段、容易遗漏验收项，或用户给了多个要求。
- todo 必须具体、可执行、可完成；同一时间只保留一个 in_progress。
- 完成一个 todo 后立即更新，不要最后批量标记。
- 测试失败、实现不完整、产物缺失、错误未解决时，不得把对应 todo 标为 completed。
- 简单单步任务不要为了形式使用 todo，直接完成。

# 工具使用纪律
- 你每个 assistant 回合必须调用一个可用工具。
- 用 chat(message=...) 与用户沟通、要求必要澄清、报告阶段性结果或最终结果。
- 只有当用户明确要结束会话时，才使用 decide(target="end", content="...")。
- 运行命令时要匹配当前 shell 和平台。当前环境如果是 PowerShell，就使用 PowerShell 写法。
- 长时间任务可用后台工具；但最终回答前必须确认后台任务完成或明确说明仍在运行。
- 命令失败后不要盲目重跑同一个命令；先读错误、检查假设、做针对性修正。
- 不使用危险命令清空、重置、删除用户工作，除非用户明确要求且范围清楚。

# 数据科学任务纪律
- 建模、清洗、分析任务必须先做真实数据画像：文件、行列数、字段类型、目标、缺失、时间顺序、泄漏风险。
- 时间序列任务默认禁止随机切分；特征、填充、标准化和评估必须避免未来信息泄漏。
- 先做可运行 baseline，再做改进；没有 baseline 的复杂模型结果不算可靠。
- 指标、图表、报告、预测表等产物必须真实生成并检查路径。
- 如果用户要求可视化，指标不能替代图；必须生成或明确说明为什么无法生成。

# 汇报规则
- 汇报要短、具体、基于证据。优先说完成了什么、验证了什么、剩余风险是什么。
- 不输出大段隐藏推理，不把工具日志完整复制给用户，只转述关键结果。
- 不夸大，不防御，不假绿。失败就是失败，未验证就是未验证，通过就是通过。
- 如果任务还没完成且可以继续修，不要把中间错误包装成最终答复；继续工作。
"""


def _codex_tools():
    return [
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
        todo_read,
        todo_write,
        chat,
        decide,
    ]


def _create_run_id() -> str:
    return "codex_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _format_run_label(run_id: str) -> str:
    stem = run_id
    for prefix in ("codex_", "vibe_", "run_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    parts = stem.split("_")
    if len(parts) == 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
        d, t = parts
        return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
    return run_id


def _list_existing_runs() -> list[tuple[str, str]]:
    if not MEM_ROOT.is_dir():
        return []

    runs: list[tuple[str, str]] = []
    for child in sorted(MEM_ROOT.iterdir()):
        if not child.is_dir():
            continue
        history = child / f"{NODE_NAME}.json"
        if not history.exists():
            continue
        try:
            records = json.loads(history.read_text(encoding="utf-8") or "[]")
        except json.JSONDecodeError:
            records = []
        record_count = len(records) if isinstance(records, list) else 0
        label = f"{_format_run_label(child.name)}  [{NODE_NAME}; {record_count} records]"
        runs.append((child.name, label))
    return runs


def _pick_run_id() -> str | None:
    runs = _list_existing_runs()
    if not runs:
        print("No previous Codex sessions found. Starting a new session.\n")
        return None

    print("Previous Codex sessions:")
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
        if 0 <= index < len(runs):
            selected = runs[index][0]
            print(f"  -> Resuming {selected}")
            return selected
    except ValueError:
        pass

    print("  -> Invalid choice, starting a new session.")
    return None


def build_codex_agent(run_id: str | None = None) -> dict:
    resolved_run_id = run_id or _create_run_id()
    platform = (os.getenv("CODEX_PLATFORM") or os.getenv("AUTOML_PLATFORM") or "openai").lower()
    model = (
        os.getenv("CODEX_MODEL")
        or os.getenv("AUTOML_MODEL")
        or os.getenv("OPENAI_MODEL")
        or ("qwen3.6" if platform in {"local", "chat", "local-responses"} else "gpt-5.5")
    )
    print(
        "[Codex LLM] "
        f"platform={platform} "
        f"model={model} "
        f"base_url={os.getenv('OPENAI_BASE_URL') or ''}",
        flush=True,
    )
    if platform in {"local", "chat"}:
        llm = local_chat_llm(
            model=model,
            reasoning_effort=os.getenv("CODEX_REASONING_EFFORT") or "medium",
            reasoning_summary=os.getenv("CODEX_REASONING_SUMMARY") or "auto",
            tool_choice="required",
        )
    elif platform == "local-responses":
        llm = local_responses_llm(
            model=model,
            reasoning_effort=os.getenv("CODEX_REASONING_EFFORT") or "medium",
            reasoning_summary=os.getenv("CODEX_REASONING_SUMMARY") or "auto",
            tool_choice="required",
        )
    else:
        llm = openai_llm(
            model=model,
            reasoning_effort=os.getenv("CODEX_REASONING_EFFORT") or "medium",
            reasoning_summary=os.getenv("CODEX_REASONING_SUMMARY") or "auto",
            tool_choice="required",
        )
    codex = node(
        name=NODE_NAME,
        display_name="Codex",
        llm=llm,
        tools=_codex_tools(),
        prompt=CODEX_PROMPT,
        edges={"end": "Use only when the user explicitly ends the session."},
    )
    return graph([codex], run_id=resolved_run_id, start_node=NODE_NAME)


def main() -> None:
    stream = "--stream" in sys.argv
    print("Codex (single generic coding agent)")
    print("=" * 68)

    run_id = _pick_run_id()
    print()

    codex_graph = build_codex_agent(run_id=run_id)
    user_message = input("Please describe your request: ").strip()
    if not user_message:
        user_message = "Inspect this workspace and help me with the next useful step."
    print()
    run(codex_graph, user_message=user_message, stream=stream)


if __name__ == "__main__":
    main()
