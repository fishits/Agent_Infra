"""FrameWork_light 文件 IO 工具。

规则：
- read_file 总是按行范围读取
- 大文件自动截断，限制字符数
- 每个响应显示文件路径、实际行范围、实际字符数、总行数、总字符数、是否截断
"""

from __future__ import annotations

import fnmatch
import json
import locale
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .range_preview import build_range_preview, extract_preview_ranges
from .todo_state import normalize_todos, read_todo_state, write_todo_state
from .tools import tool


MAX_READ_CHARS = 8000           #自动阶段最大字符数，防止IO爆炸
MAX_FOREGROUND_DISPLAY_CHARS = 4000
MAX_READ_FILES = 5              #read_many最大文件数，防止IO爆炸
READ_MANY_PREVIEW_LINES = 200   #read_many每个文件的预览行数，超过后自动截断
MAX_GREP_RESULTS = 20
MAX_LIST_ENTRIES = 100          #list_directory最大条目数，超过后自动截断
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv"}
VISIBLE_DOT_DIRS = {".automl"}
MAX_GREP_RESULTS = 20
MAX_PARALLEL_TOOL_TASKS = 8
MAX_PARALLEL_RESULT_CHARS = 12000
DEFAULT_BACKGROUND_DIR = Path("coding") / "background_jobs"
DEFAULT_POLL_INTERVAL_SECONDS = 10
MAX_FOREGROUND_EXECUTION_SECONDS = 30
INITIAL_BACKGROUND_WAIT_SECONDS = 0
_CURRENT_TODOS: list[dict[str, str]] = []


@tool(
    description=(
        "按行号范围读取文件。path、start_line、end_line 都支持默认值。"
        "结果会以范围信息返回；如果需要继续读取，请使用返回的后续行范围。"
    ),
    params={
        "path": ("string", "必须显式填写：相对于当前工作目录的文件路径"),
        "start_line": ("integer", "必须显式填写：1-based 起始行号"),
        "end_line": ("integer", "必须显式填写：1-based 结束行号"),
    },
)
def read_file(
    path: str,
    start_line: int = 1,
    end_line: int = 200,
    runtime_context: dict[str, Any] | None = None,
) -> str:
    try:
        target = _resolve_path(path, runtime_context)
    except ValueError as exc:
        return str(exc)
    if not target.exists():
        return "文件不存在"
    if not target.is_file():
        return "目标不是文件"
    if start_line < 1 or end_line < start_line:
        return "读取范围错误"

    text = target.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    total_chars = len(text)

    if total_lines == 0:
        return _format_read_result(
            path=path,
            start_line=1,
            end_line=0,
            total_lines=0,
            content="",
            total_chars=0,
            next_line_range=None,
        )

    start_index = min(start_line - 1, total_lines)
    requested_end_index = min(end_line, total_lines)
    selected_lines = lines[start_index:requested_end_index]

    content, actual_end_line, truncated = _truncate_lines(
        selected_lines=selected_lines,
        start_line=start_line,
        requested_end_line=requested_end_index,
        total_lines=total_lines,
        char_limit=MAX_READ_CHARS,
    )

    next_start_line = actual_end_line + 1 if truncated and actual_end_line < total_lines else None
    return _format_read_result(
        path=path,
        start_line=start_line,
        end_line=actual_end_line,
        total_lines=total_lines,
        content=content,
        total_chars=total_chars,
        next_line_range=(next_start_line, total_lines) if next_start_line is not None else None,
    )


@tool(
    description=(
        "一次读取多个文件的预览。用于 glob/search 后比较少量候选文件。"
        "如果需要混合读取、搜索、列目录，优先使用 parallel_tools。"
    ),
    params={
        "paths": ("array", "文件路径列表，相对于当前工作目录"),
    },
)
def read_many(paths: list[str], runtime_context: dict[str, Any] | None = None) -> str:
    if not paths:
        return "缺少文件路径"

    selected_paths = paths[:MAX_READ_FILES]
    blocks = []
    if paths:
        blocks.append(f"# files: {len(selected_paths)}/{len(paths)}")
    if len(paths) > MAX_READ_FILES:
        blocks.append(f"# remaining_files: {len(selected_paths) + 1}-{len(paths)}/{len(paths)}")
        blocks.append("")

    for item in selected_paths:
        blocks.append(read_file(item, 1, READ_MANY_PREVIEW_LINES, runtime_context=runtime_context))
        blocks.append("")

    return "\n".join(blocks).rstrip()


@tool(
    description=(
        "按深度列出目录结构，用于理解项目布局。"
        "按文件名模式查找文件时使用 glob_files。"
    ),
    params={
        "path": ("string", "目录路径，相对于当前工作目录"),
        "depth": ("integer", "最大递归深度"),
    },
)
def list_directory(path: str = ".", depth: int = 2, runtime_context: dict[str, Any] | None = None) -> str:
    try:
        root = _resolve_path(path, runtime_context)
    except ValueError as exc:
        return str(exc)
    if not root.exists():
        return "目录不存在"
    if not root.is_dir():
        return "目标不是目录"
    if depth < 1:
        return "目录层级错误"

    lines: list[str] = [f"# directory: {path}"]
    shown_count = 0
    total_count = 0

    def walk(current: Path, current_depth: int, prefix: str) -> None:
        nonlocal shown_count, total_count
        if current_depth > depth:
            return
        items = sorted(current.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        items = [
            item
            for item in items
            if item.name not in SKIP_DIRS and (not item.name.startswith(".") or item.name in VISIBLE_DOT_DIRS)
        ]
        for item in items:
            total_count += 1
            label = item.name + ("/" if item.is_dir() else "")
            if shown_count < MAX_LIST_ENTRIES:
                lines.append(f"{prefix}{label}")
                shown_count += 1
            if item.is_dir():
                walk(item, current_depth + 1, prefix + "  ")

    walk(root, 1, "")
    lines.append(f"# entries: {shown_count}/{total_count}")
    return "\n".join(lines)


@tool(
    description=(
        "按 glob 模式快速查找文件，例如 **/*.py 或 src/**/*.ts。"
        "返回按修改时间排序的文件路径；查找文件时优先用这个工具，不要通过 PowerShell 搜索。"
    ),
    params={
        "pattern": ("string", "相对于当前工作目录的 glob 模式"),
        "max_results": ("integer", "最多返回的文件路径数量"),
    },
)
def glob_files(pattern: str, max_results: int = 100, runtime_context: dict[str, Any] | None = None) -> str:
    if not pattern.strip():
        return "缺少文件模式"
    if max_results < 1:
        return "结果数量错误"

    root = _workspace_root(runtime_context)
    matches: list[Path] = []
    for file_path in root.glob(pattern):
        if not file_path.is_file():
            continue
        try:
            relative = file_path.relative_to(root)
        except ValueError:
            continue
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        matches.append(file_path)

    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    selected = matches[:max_results]
    lines = [
        f"# pattern: {pattern}",
        f"# matches: {len(matches)}",
        f"# returned: {len(selected)}/{len(matches)}",
        "",
    ]
    lines.extend(str(path.relative_to(root)) for path in selected)
    return "\n".join(lines).rstrip()


@tool(
    description=(
        "在当前工作目录下搜索文本或正则。代码/内容搜索优先用这个工具，不要通过 PowerShell 调 rg、grep 或 Select-String。"
    ),
    params={
        "query": ("string", "要搜索的文本或正则表达式"),
        "include_pattern": ("string", "文件 glob 范围，例如 **/*.py"),
        "max_results": ("integer", "最多返回的匹配数量"),
    },
)
def grep_search(
    query: str,
    include_pattern: str = "**/*",
    max_results: int = MAX_GREP_RESULTS,
    runtime_context: dict[str, Any] | None = None,
) -> str:
    if not query.strip():
        return "缺少搜索内容"
    if max_results < 1:
        return "结果数量错误"

    root = _workspace_root(runtime_context)
    try:
        pattern = re.compile(query, re.IGNORECASE)
    except re.error:
        return "搜索表达式错误"

    results: list[str] = []
    total_matches = 0
    for file_path in root.rglob("*"):
        if not file_path.is_file():
            continue
        if any(part in SKIP_DIRS for part in file_path.parts):
            continue
        relative_path = file_path.relative_to(root)
        if not fnmatch.fnmatch(str(relative_path), include_pattern):
            continue
        try:
            for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                if pattern.search(line):
                    total_matches += 1
                    if len(results) < max_results:
                        results.append(f"{relative_path}:{line_number}: {line}")
        except (UnicodeDecodeError, PermissionError):
            continue

    return _format_grep_result(query, results, total_matches)


@tool(
    description=(
        "并行执行多个彼此独立的读取、搜索、列目录任务，并按提交顺序返回结果。"
        "需要同时检查多个文件、多个搜索或混合仓库信息时使用。"
        "不要用于写入、补丁、shell 命令或有依赖顺序的任务。"
    ),
    params={
        "tasks_json": ("string", "JSON 数组，每项为 {tool, args}。允许工具：read_file、read_many、list_directory、glob_files、grep_search、todo_read"),
    },
)
def parallel_tools(tasks_json: str, runtime_context: dict[str, Any] | None = None) -> str:
    try:
        tasks = json.loads(tasks_json)
    except json.JSONDecodeError:
        return "tasks_json 必须是 JSON 数组"
    if not isinstance(tasks, list):
        return "tasks_json 必须是 JSON 数组"
    if len(tasks) > MAX_PARALLEL_TOOL_TASKS:
        return f"并行任务过多：最多 {MAX_PARALLEL_TOOL_TASKS} 个"

    allowed = {
        "read_file": read_file,
        "read_many": read_many,
        "list_directory": list_directory,
        "glob_files": glob_files,
        "grep_search": grep_search,
        "todo_read": todo_read,
    }

    def run_one(index: int, task: Any) -> dict[str, Any]:
        if not isinstance(task, dict):
            return {"index": index, "ok": False, "error": "task 必须是对象"}
        tool_name = str(task.get("tool") or "")
        args = task.get("args") or {}
        if tool_name not in allowed:
            return {"index": index, "tool": tool_name, "ok": False, "error": f"不允许调用该工具: {tool_name}"}
        if not isinstance(args, dict):
            return {"index": index, "tool": tool_name, "ok": False, "error": "args 必须是对象"}
        try:
            if tool_name == "todo_read":
                output = allowed[tool_name](runtime_context=runtime_context or {})
            else:
                output = allowed[tool_name](**args)
            return {"index": index, "tool": tool_name, "ok": True, "output": str(output)}
        except Exception as exc:
            return {"index": index, "tool": tool_name, "ok": False, "error": str(exc)}

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(len(tasks) or 1, MAX_PARALLEL_TOOL_TASKS)) as executor:
        futures = [executor.submit(run_one, index, task) for index, task in enumerate(tasks)]
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item["index"])
    output = json.dumps({"results": results}, ensure_ascii=False, indent=2)
    if len(output) > MAX_PARALLEL_RESULT_CHARS:
        output = build_range_preview(output, MAX_PARALLEL_RESULT_CHARS)
    return output


@tool(
    description=(
        "写入完整文件。编辑已有文件时优先使用 apply_patch 或 apply_diff。"
        "仅在创建新文件或明确要整体替换文件时使用。"
    ),
    params={
        "path": ("string", "目标文件路径，相对于当前工作目录"),
        "content": ("string", "完整文件内容"),
    },
)
def write_file(path: str, content: str, runtime_context: dict[str, Any] | None = None) -> str:
    try:
        target = _resolve_path(path, runtime_context)
    except ValueError as exc:
        return str(exc)
    target.parent.mkdir(parents=True, exist_ok=True)
    requested_chars = len(content)
    requested_lines = content.count("\n") + (0 if not content else 1)
    written_content = content[:MAX_READ_CHARS]
    target.write_text(written_content, encoding="utf-8")
    written_chars = len(written_content)
    written_lines = written_content.count("\n") + (0 if not written_content else 1)

    if requested_chars <= MAX_READ_CHARS:
        return (
            f"# path: {path}\n"
            f"# chars: {written_chars}/{written_chars}\n"
            f"# lines: 1-{written_lines}/{written_lines}\n\n"
            f"文件已写入"
        )

    tail_content, preview_start_line, preview_end_line = _tail_preview_for_write(
        written_content,
        preview_char_limit=min(1600, MAX_READ_CHARS),
    )
    preview_chars = len(tail_content)
    return (
        f"# path: {path}\n"
        f"# requested_chars: {requested_chars}\n"
        f"# requested_lines: {requested_lines}\n"
        f"# current_file_chars: {written_chars}/{written_chars}\n"
        f"# current_file_lines: 1-{written_lines}/{written_lines}\n\n"
        "单次输出过长已被截断\n"
        "已将当前可写入的内容直接写入目标文件。\n\n"
        f"# preview_chars: {preview_chars}/{written_chars}\n"
        f"# preview_lines: {preview_start_line}-{preview_end_line}/{written_lines}\n\n"
        f"{tail_content}\n\n"
        "请使用 apply_patch、append_file 或再次调用 write_file 继续完成书写。"
    )


@tool(
    description=(
        "向文件末尾追加内容。仅在明确需要追加时使用，例如日志或追加配置块。"
        "普通代码编辑优先使用 apply_patch 或 apply_diff。"
    ),
    params={
        "path": ("string", "目标文件路径，相对于当前工作目录"),
        "content": ("string", "要追加到文件末尾的内容"),
    },
)
def append_file(path: str, content: str, runtime_context: dict[str, Any] | None = None) -> str:
    try:
        target = _resolve_path(path, runtime_context)
    except ValueError as exc:
        return str(exc)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(content)
    appended_lines = content.count("\n") + (0 if not content else 1)
    return f"append ok: {path} ({appended_lines} lines, {len(content)} chars)"


def _tail_preview_for_write(text: str, preview_char_limit: int) -> tuple[str, int, int]:
    if not text:
        return "", 0, 0

    lines = text.splitlines(keepends=True)
    total_lines = len(lines)
    selected: list[str] = []
    selected_chars = 0

    for line in reversed(lines):
        if selected and selected_chars + len(line) > preview_char_limit:
            break
        selected.append(line)
        selected_chars += len(line)

    selected.reverse()
    preview = "".join(selected)
    preview_line_count = len(selected)
    start_line = max(1, total_lines - preview_line_count + 1)
    end_line = total_lines
    return preview.rstrip(), start_line, end_line


@tool(
    description="已禁用。本项目不允许执行 Bash。",
    params={
        "command": ("string", "要执行的 Bash 命令"),
        "timeout": ("integer", "超时时间，单位秒"),
    },
)
def run_bash(command: str, timeout: int = 30) -> str:
    return "run_bash 已禁用，请改用 run_powershell。"


@tool(
    description=(
        "执行 Python 代码。最多前台等待 30 秒；仍在运行则返回后台任务 pid/log_path。"
    ),
    params={
        "code": ("string", "要执行的 Python 代码"),
        "timeout": ("integer", "超时时间，单位秒"),
    },
)
def run_python(code: str, timeout: int = 30, runtime_context: dict[str, Any] | None = None) -> str:
    if not code.strip():
        return "缺少 Python 代码"
    try:
        workspace_root = _workspace_root(runtime_context)
        script_path = _write_background_python_script(code, runtime_context)
        log_path = _default_background_log_path("python", runtime_context)
        process = _start_logged_process(
            command=["python", "-u", str(script_path)],
            log_path=log_path,
            command_text=f"python -u {_path_for_display(script_path, runtime_context)}",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            cwd=workspace_root,
        )
        return _wait_or_background_result(
            process=process,
            log_path=log_path,
            command=f"python -u {_path_for_display(script_path, runtime_context)}",
            requested_timeout=timeout,
            kind="python",
            extra_metadata={"script_path": _path_for_display(script_path, runtime_context)},
            runtime_context=runtime_context,
        )
    except FileNotFoundError:
        return "未找到 python，请确认 Python 已安装并在 PATH 中。"
    except Exception as exc:
        return f"execution failed: {exc}"


@tool(
    description=(
        "执行 PowerShell 命令。最多前台等待 30 秒；仍在运行则返回后台任务 pid/log_path。"
    ),
    params={
        "command": ("string", "要执行的 PowerShell 命令"),
        "timeout": ("integer", "超时时间，单位秒"),
    },
)
def run_powershell(command: str, timeout: int = 30, runtime_context: dict[str, Any] | None = None) -> str:
    guard = _powershell_guardrail(command)
    if guard is not None:
        return guard
    try:
        workspace_root = _workspace_root(runtime_context)
        log_path = _default_background_log_path("powershell", runtime_context)
        process = _start_logged_process(
            command=["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            log_path=log_path,
            command_text=command,
            cwd=workspace_root,
        )
        return _wait_or_background_result(
            process=process,
            log_path=log_path,
            command=command,
            requested_timeout=timeout,
            kind="powershell",
            runtime_context=runtime_context,
        )
    except FileNotFoundError:
        return "PowerShell 未找到，请确认系统已安装 PowerShell"
    except Exception as exc:
        return f"执行失败: {exc}"


@tool(
    description=(
        "在后台启动长时间运行的 PowerShell 命令，并把 stdout/stderr 写入日志文件。"
        "仅在命令可能运行很久且不需要立刻拿到输出时使用。"
    ),
    params={
        "command": ("string", "要在后台启动的 PowerShell 命令"),
        "log_path": ("string", "日志文件路径，相对于当前工作目录"),
    },
)
def start_background_powershell(
    command: str,
    log_path: str,
    runtime_context: dict[str, Any] | None = None,
) -> str:
    try:
        target_log = _resolve_path(log_path, runtime_context)
    except ValueError as exc:
        return str(exc)

    workspace_root = _workspace_root(runtime_context)
    process = _start_logged_process(
        command=["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
        log_path=target_log,
        command_text=command,
        cwd=workspace_root,
    )
    initial_status = _process_status(process.pid)

    metadata = {
        "pid": process.pid,
        "command": command,
        "log_path": _path_for_display(target_log, runtime_context),
        "started_at": _now_text(),
        "poll_interval_seconds": DEFAULT_POLL_INTERVAL_SECONDS,
        "initial_status": initial_status,
        "initial_wait_seconds": INITIAL_BACKGROUND_WAIT_SECONDS,
    }
    metadata_path = _background_metadata_path(process.pid, runtime_context)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.dumps(metadata, ensure_ascii=False, indent=2)


@tool(
    description=(
        "检查后台任务是否仍在运行，并预览日志末尾。"
    ),
    params={
        "pid": ("integer", "start_background_powershell 返回的进程 id"),
        "log_path": ("string", "日志文件路径，相对于当前工作目录"),
        "tail_lines": ("integer", "要预览的日志尾部行数"),
    },
)
def check_background_powershell(
    pid: int,
    log_path: str,
    tail_lines: int = 40,
    runtime_context: dict[str, Any] | None = None,
) -> str:
    try:
        target_log = _resolve_path(log_path, runtime_context)
    except ValueError as exc:
        return str(exc)

    safe_tail_lines = max(1, min(tail_lines, 200))
    tail_text = _read_log_tail(target_log, safe_tail_lines)
    payload = {
        "pid": pid,
        "status": _process_status(pid),
        "log_path": log_path,
        "tail_lines": safe_tail_lines,
        "recommended_sleep_seconds": DEFAULT_POLL_INTERVAL_SECONDS,
        "log_tail": tail_text,
        "log_tail_line_count": 0 if tail_text.startswith("(") else len(tail_text.splitlines()),
    }
    if payload["status"] == "finished":
        payload["recommended_sleep_seconds"] = 0
    return json.dumps(payload, ensure_ascii=False, indent=2)


@tool(
    description=(
        "立即检查后台任务。为了避免阻塞，本工具不会实际 sleep。"
    ),
    params={
        "pid": ("integer", "start_background_powershell 返回的进程 id"),
        "log_path": ("string", "日志文件路径，相对于当前工作目录"),
        "seconds": ("integer", "检查前等待的秒数"),
        "tail_lines": ("integer", "要预览的日志尾部行数"),
    },
)
def sleep_then_check_background_powershell(
    pid: int,
    log_path: str,
    seconds: int = DEFAULT_POLL_INTERVAL_SECONDS,
    tail_lines: int = 40,
    runtime_context: dict[str, Any] | None = None,
) -> str:
    requested_seconds = max(0, seconds)
    safe_seconds = min(requested_seconds, 300)
    if safe_seconds:
        time.sleep(safe_seconds)
    status_json = check_background_powershell(
        pid=pid,
        log_path=log_path,
        tail_lines=tail_lines,
        runtime_context=runtime_context,
    )
    return json.dumps(
        {
            "requested_wait_seconds": requested_seconds,
            "waited_seconds": safe_seconds,
            "status_after_wait": json.loads(status_json),
        },
        ensure_ascii=False,
        indent=2,
    )


@tool(
    description=(
        "在文件内执行一次精确文本替换。已知完整源文本且只做小编辑时使用。"
        "search 必须精确匹配；默认要求唯一匹配，只有明确要全文件替换时才设置 replace_all=true。"
        "多个 hunk 或多个文件请使用 apply_patch。"
    ),
    params={
        "path": ("string", "目标文件路径，相对于当前工作目录"),
        "search": ("string", "要查找的精确源文本"),
        "replace": ("string", "替换后的文本"),
        "replace_all": ("boolean", "是否替换全部匹配；false 时要求恰好一个匹配"),
    },
)
def apply_diff(
    path: str,
    search: str,
    replace: str,
    replace_all: bool = False,
    runtime_context: dict[str, Any] | None = None,
) -> str:
    try:
        target = _resolve_path(path, runtime_context)
    except ValueError as exc:
        return str(exc)
    if not target.exists():
        return "文件不存在"
    if not target.is_file():
        return "目标不是文件"
    if not search:
        return "缺少搜索内容"

    content = target.read_text(encoding="utf-8")
    match_count = content.count(search)
    if match_count == 0:
        return (
            f"编辑失败：在 {path} 中没有找到 search 文本。"
            "请先读取当前文件内容，并使用精确片段。"
        )
    if match_count > 1 and not replace_all:
        return (
            f"编辑失败：search 文本在 {path} 中不唯一（{match_count} 处匹配）。"
            "请增加上下文，或在确实要改全部匹配时设置 replace_all=true。"
        )

    updated = content.replace(search, replace) if replace_all else content.replace(search, replace, 1)
    target.write_text(updated, encoding="utf-8")
    return f"文件已更新: {path} ({match_count if replace_all else 1} 处替换)"


@tool(
    description=(
        "应用 Codex 风格的多文件补丁，格式为 *** Begin Patch。"
        "支持 Add File、Delete File、Update File、可选 Move to 和多个 hunk。"
        "必须使用精确上下文；歧义更新会失败，不会猜测。"
    ),
    params={
        "patch": ("string", "以 *** Begin Patch 开始、以 *** End Patch 结束的补丁文本"),
    },
)
def apply_patch(patch: str) -> str:
    try:
        operations = _parse_apply_patch_v2(patch)
        changed = _apply_patch_operations_v2(operations)
    except ValueError as exc:
        return str(exc)
    return "补丁已应用:\n" + "\n".join(f"- {item}" for item in changed)


@tool(
    description="读取当前编码会话的持久 todo 列表。",
    params={},
)
def todo_read(runtime_context: dict[str, Any] | None = None) -> str:
    context = runtime_context or {}
    todos = read_todo_state(context.get("run_id"), context.get("node_name"))
    if not todos:
        return "todo 列表为空"
    lines = ["当前 todos:"]
    for index, item in enumerate(todos, start=1):
        lines.append(f"{index}. [{item['status']}] {item['content']}")
    return "\n".join(lines)


@tool(
    description=(
        "更新当前编码会话的 todo 列表。非平凡多步骤任务、收到多条需求后、进度变化时主动使用。"
        "必须保持恰好一个 in_progress 项。"
        "每项必须同时提供 content 和 activeForm。"
    ),
    params={
        "todos_json": ("string", "JSON 数组，每项为 {content, status, activeForm}。status 只能是 pending、in_progress 或 completed"),
    },
)
def todo_write(todos_json: str, runtime_context: dict[str, Any] | None = None) -> str:
    global _CURRENT_TODOS
    try:
        payload = json.loads(todos_json)
    except json.JSONDecodeError:
        return "todos_json 必须是 JSON 数组"

    normalized, error = normalize_todos(payload)
    if error or normalized is None:
        return error or "todo 列表无效"

    _CURRENT_TODOS = normalized
    context = runtime_context or {}
    write_todo_state(context.get("run_id"), context.get("node_name"), normalized)
    lines = ["todos updated:"]
    for index, item in enumerate(_CURRENT_TODOS, start=1):
        lines.append(f"{index}. [{item['status']}] {item['content']}")
    return "\n".join(lines)


def _truncate_lines(
    *,
    selected_lines: list[str],
    start_line: int,
    requested_end_line: int,
    total_lines: int,
    char_limit: int,
) -> tuple[str, int, bool]:
    if not selected_lines:
        return "", min(start_line - 1, total_lines), False

    parts: list[str] = []
    current_chars = 0
    actual_end_line = start_line - 1
    truncated = False

    for offset, line in enumerate(selected_lines):
        next_chars = current_chars + len(line)
        if parts and next_chars > char_limit:
            truncated = True
            break
        if not parts and len(line) > char_limit:
            parts.append(line[:char_limit])
            actual_end_line = start_line + offset
            truncated = True
            break

        parts.append(line)
        current_chars = next_chars
        actual_end_line = start_line + offset

    if actual_end_line < requested_end_line:
        truncated = truncated or actual_end_line < min(requested_end_line, total_lines)

    return "".join(parts), actual_end_line, truncated


def _parse_apply_patch(patch: str) -> list[dict[str, Any]]:
    lines = patch.splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("patch must start with *** Begin Patch")
    if lines[-1].strip() != "*** End Patch":
        raise ValueError("patch must end with *** End Patch")

    operations: list[dict[str, Any]] = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if line.startswith("*** Add File: "):
            path = line.removeprefix("*** Add File: ").strip()
            index += 1
            content: list[str] = []
            while index < len(lines) - 1 and not lines[index].startswith("*** "):
                if not lines[index].startswith("+"):
                    raise ValueError("Add File lines must start with +")
                content.append(lines[index][1:])
                index += 1
            operations.append({"op": "add", "path": path, "content": "\n".join(content) + ("\n" if content else "")})
            continue

        if line.startswith("*** Delete File: "):
            path = line.removeprefix("*** Delete File: ").strip()
            operations.append({"op": "delete", "path": path})
            index += 1
            continue

        if line.startswith("*** Update File: "):
            path = line.removeprefix("*** Update File: ").strip()
            index += 1
            hunks: list[list[str]] = []
            current: list[str] = []
            while index < len(lines) - 1 and not lines[index].startswith("*** "):
                hunk_line = lines[index]
                if hunk_line.startswith("@@"):
                    if current:
                        hunks.append(current)
                        current = []
                    index += 1
                    continue
                if hunk_line == "*** End of File":
                    index += 1
                    continue
                if not hunk_line or hunk_line[0] not in {" ", "+", "-"}:
                    raise ValueError(f"invalid patch line: {hunk_line}")
                current.append(hunk_line)
                index += 1
            if current:
                hunks.append(current)
            if not hunks:
                raise ValueError(f"Update File has no hunks: {path}")
            operations.append({"op": "update", "path": path, "hunks": hunks})
            continue

        if not line.strip():
            index += 1
            continue
        raise ValueError(f"unknown patch directive: {line}")

    return operations


def _parse_apply_patch_v2(patch: str) -> list[dict[str, Any]]:
    lines = patch.splitlines()
    if not lines or lines[0].strip() != "*** Begin Patch":
        raise ValueError("patch must start with *** Begin Patch")
    if lines[-1].strip() != "*** End Patch":
        raise ValueError("patch must end with *** End Patch")

    operations: list[dict[str, Any]] = []
    index = 1
    while index < len(lines) - 1:
        line = lines[index]
        if not line.strip():
            index += 1
            continue

        if line.startswith("*** Add File: "):
            path = line.removeprefix("*** Add File: ").strip()
            if not path:
                raise ValueError("Add File path is required")
            index += 1
            content: list[str] = []
            while index < len(lines) - 1 and not lines[index].startswith("*** "):
                if not lines[index].startswith("+"):
                    raise ValueError(f"Add File lines must start with +: {path}")
                content.append(lines[index][1:])
                index += 1
            operations.append({"op": "add", "path": path, "content": "\n".join(content) + ("\n" if content else "")})
            continue

        if line.startswith("*** Delete File: "):
            path = line.removeprefix("*** Delete File: ").strip()
            if not path:
                raise ValueError("Delete File path is required")
            operations.append({"op": "delete", "path": path})
            index += 1
            continue

        if line.startswith("*** Update File: "):
            path = line.removeprefix("*** Update File: ").strip()
            if not path:
                raise ValueError("Update File path is required")
            index += 1
            move_to = None
            if index < len(lines) - 1 and lines[index].startswith("*** Move to: "):
                move_to = lines[index].removeprefix("*** Move to: ").strip()
                if not move_to:
                    raise ValueError(f"Move to path is required: {path}")
                index += 1

            hunks: list[list[str]] = []
            current: list[str] = []
            while index < len(lines) - 1 and not lines[index].startswith("*** "):
                hunk_line = lines[index]
                if hunk_line.startswith("@@"):
                    if current:
                        hunks.append(current)
                        current = []
                    index += 1
                    continue
                if hunk_line == "*** End of File":
                    index += 1
                    continue
                if not hunk_line or hunk_line[0] not in {" ", "+", "-"}:
                    raise ValueError(f"invalid patch line in {path}: {hunk_line}")
                current.append(hunk_line)
                index += 1
            if current:
                hunks.append(current)
            if not hunks:
                raise ValueError(f"Update File has no hunks: {path}")
            operations.append({"op": "update", "path": path, "move_to": move_to, "hunks": hunks})
            continue

        raise ValueError(f"unknown patch directive: {line}")

    if not operations:
        raise ValueError("patch contains no operations")
    return operations


def _apply_patch_operations_v2(operations: list[dict[str, Any]]) -> list[str]:
    pending: dict[Path, str | None] = {}
    changed: list[str] = []

    for operation in operations:
        target = _resolve_path(operation["path"])
        op = operation["op"]

        if op == "add":
            if target.exists() or (target in pending and pending[target] is not None):
                raise ValueError(f"file already exists: {operation['path']}")
            pending[target] = operation["content"]
            changed.append(f"added {operation['path']}")
            continue

        if op == "delete":
            if target in pending:
                if pending[target] is None:
                    raise ValueError(f"file already deleted: {operation['path']}")
                pending[target] = None
                changed.append(f"deleted {operation['path']}")
                continue
            if not target.exists():
                raise ValueError(f"file does not exist: {operation['path']}")
            if not target.is_file():
                raise ValueError(f"target is not a file: {operation['path']}")
            pending[target] = None
            changed.append(f"deleted {operation['path']}")
            continue

        if op == "update":
            if target in pending:
                content = pending[target]
                if content is None:
                    raise ValueError(f"file already deleted: {operation['path']}")
            elif target.exists():
                if not target.is_file():
                    raise ValueError(f"target is not a file: {operation['path']}")
                content = target.read_text(encoding="utf-8")
            else:
                raise ValueError(f"file does not exist: {operation['path']}")

            for hunk in operation["hunks"]:
                old, new = _hunk_to_blocks(hunk)
                if not old:
                    content += new
                    continue
                match_count = content.count(old)
                if match_count == 0:
                    raise ValueError(f"patch context not found: {operation['path']}")
                if match_count > 1:
                    raise ValueError(f"patch context is not unique: {operation['path']}")
                content = content.replace(old, new, 1)

            move_to = operation.get("move_to")
            if move_to:
                move_target = _resolve_path(move_to)
                if move_target.exists() or (move_target in pending and pending[move_target] is not None):
                    raise ValueError(f"move target already exists: {move_to}")
                pending[target] = None
                pending[move_target] = content
                changed.append(f"updated {operation['path']} -> {move_to}")
            else:
                pending[target] = content
                changed.append(f"updated {operation['path']}")
            continue

        raise ValueError(f"unknown patch operation: {op}")

    for target, content in pending.items():
        if content is None:
            if target.exists():
                target.unlink()
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    return changed


def _apply_patch_operations(operations: list[dict[str, Any]]) -> list[str]:
    changed: list[str] = []
    for operation in operations:
        target = _resolve_path(operation["path"])
        op = operation["op"]

        if op == "add":
            if target.exists():
                raise ValueError(f"file already exists: {operation['path']}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(operation["content"], encoding="utf-8")
            changed.append(f"added {operation['path']}")
            continue

        if op == "delete":
            if not target.exists():
                raise ValueError(f"file does not exist: {operation['path']}")
            if not target.is_file():
                raise ValueError(f"target is not a file: {operation['path']}")
            target.unlink()
            changed.append(f"deleted {operation['path']}")
            continue

        if op == "update":
            if not target.exists():
                raise ValueError(f"file does not exist: {operation['path']}")
            content = target.read_text(encoding="utf-8")
            for hunk in operation["hunks"]:
                old, new = _hunk_to_blocks(hunk)
                if old not in content:
                    raise ValueError(f"patch context not found: {operation['path']}")
                content = content.replace(old, new, 1)
            target.write_text(content, encoding="utf-8")
            changed.append(f"updated {operation['path']}")
            continue

    return changed


def _hunk_to_blocks(hunk: list[str]) -> tuple[str, str]:
    old_parts: list[str] = []
    new_parts: list[str] = []
    for line in hunk:
        marker = line[0]
        body = line[1:]
        if marker in {" ", "-"}:
            old_parts.append(body + "\n")
        if marker in {" ", "+"}:
            new_parts.append(body + "\n")
    return "".join(old_parts), "".join(new_parts)


def _format_read_result(
    *,
    path: str,
    start_line: int,
    end_line: int,
    total_lines: int,
    content: str,
    total_chars: int,
    next_line_range: tuple[int, int] | None,
) -> str:
    lines = [
        f"# file: {path}",
        f"# lines: {start_line}-{end_line}/{total_lines}",
        f"# chars: {len(content)}/{total_chars}",
    ]
    if next_line_range is not None:
        next_start, next_end = next_line_range
        lines.append(f"# next_lines: {next_start}-{next_end}/{total_lines}")
    lines.append("")
    lines.append(content)
    return "\n".join(lines).rstrip()


def _format_grep_result(query: str, results: list[str], total_matches: int) -> str:
    lines = [
        f"# query: {query}",
        f"# matches: {len(results)}/{total_matches}",
        "",
    ]
    lines.extend(results or ["未找到匹配内容"])
    return "\n".join(lines).rstrip()


def _write_background_python_script(code: str, runtime_context: dict[str, Any] | None = None) -> Path:
    background_dir = _background_dir(runtime_context)
    background_dir.mkdir(parents=True, exist_ok=True)
    script_path = background_dir / f"python_{int(time.time() * 1000)}.py"
    prelude = (
        "import sys\n"
        "try:\n"
        "    sys.stdout.reconfigure(line_buffering=True)\n"
        "    sys.stderr.reconfigure(line_buffering=True)\n"
        "except Exception:\n"
        "    pass\n\n"
    )
    script_path.write_text(prelude + code, encoding="utf-8")
    return script_path.resolve()


def _default_background_log_path(kind: str, runtime_context: dict[str, Any] | None = None) -> Path:
    background_dir = _background_dir(runtime_context)
    background_dir.mkdir(parents=True, exist_ok=True)
    return (background_dir / f"{kind}_{int(time.time() * 1000)}.log").resolve()


def _start_logged_process(
    *,
    command: list[str],
    log_path: Path,
    command_text: str,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch(exist_ok=True)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_path.open("a", encoding="utf-8", errors="replace") as stream:
        stream.write(f"\n[{_now_text()}] [launcher] starting background command\n")
        stream.write(f"[launcher] command: {command_text}\n")
        stream.flush()
        return subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=cwd or Path.cwd(),
            creationflags=creationflags,
            env=env,
        )


def _wait_or_background_result(
    *,
    process: subprocess.Popen,
    log_path: Path,
    command: str,
    requested_timeout: int,
    kind: str,
    extra_metadata: dict[str, str] | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> str:
    wait_seconds = max(0, min(int(requested_timeout or 0), MAX_FOREGROUND_EXECUTION_SECONDS))
    try:
        returncode = process.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        metadata: dict[str, Any] = {
            "status": "running",
            "pid": process.pid,
            "kind": kind,
            "command": command,
            "log_path": _path_for_display(log_path, runtime_context),
            "foreground_wait_seconds": wait_seconds,
            "recommended_tool": "check_background_powershell",
            "recommended_sleep_seconds": DEFAULT_POLL_INTERVAL_SECONDS,
            "message": "任务仍在后台运行；不要重复启动同一任务，请用 pid/log_path 检查进度。",
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        _write_background_metadata(process.pid, metadata, runtime_context)
        return json.dumps(metadata, ensure_ascii=False, indent=2)

    return _format_logged_process_result(returncode, log_path, runtime_context)


def _format_logged_process_result(
    returncode: int,
    log_path: Path,
    runtime_context: dict[str, Any] | None = None,
) -> str:
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    display_log = log_text
    if len(display_log) > MAX_FOREGROUND_DISPLAY_CHARS:
        if returncode != 0:
            display_log = _tail_chars(display_log, MAX_FOREGROUND_DISPLAY_CHARS)
        else:
            display_log = build_range_preview(
                display_log,
                MAX_FOREGROUND_DISPLAY_CHARS,
                **extract_preview_ranges(display_log),
            )
    output = (
        f"exit_code: {returncode}\n"
        f"[log_path]\n{_path_for_display(log_path, runtime_context)}\n"
        f"[stdout/stderr]\n{display_log}"
    )
    if not log_text.strip():
        output += "\n(无输出)"
    return output


def _write_background_metadata(
    pid: int,
    metadata: dict[str, Any],
    runtime_context: dict[str, Any] | None = None,
) -> None:
    metadata_path = _background_metadata_path(pid, runtime_context)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _background_metadata_path(pid: int, runtime_context: dict[str, Any] | None = None) -> Path:
    return _background_dir(runtime_context) / f"job_{pid}.json"


def _process_status(pid: int) -> str:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"if (Get-Process -Id {pid} -ErrorAction SilentlyContinue) {{ 'running' }} else {{ 'finished' }}",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    status = (result.stdout or "").strip().lower()
    return status if status in {"running", "finished"} else "unknown"


def _read_log_tail(path: Path, tail_lines: int) -> str:
    if not path.exists():
        return "(日志文件不存在)"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return "(日志为空)"
    return "\n".join(lines[-tail_lines:])


def _now_text() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _powershell_guardrail(command: str) -> str | None:
    if re.search(r"\bpython(?:\.exe)?\s+-\s*<<", command):
        return (
            "PowerShell 不支持 bash heredoc，例如 python - <<'PY'。"
            "多行 Python 请使用 run_python(code=...)。"
        )
    if re.search(r"(^|[;&|]\s*)rg(\.exe)?\b", command.strip(), flags=re.IGNORECASE):
        return (
            "当前环境不可用 rg。代码/文本搜索请使用 grep_search，"
            "文件发现请使用 glob_files。"
        )
    return None


def _format_command_result(returncode: int, stdout: bytes, stderr: bytes) -> str:
    stdout_text = _decode_output(stdout, prefer_utf8=False)
    stderr_text = _decode_output(stderr, prefer_utf8=False)
    parts = [f"exit_code: {returncode}"]
    if stderr_text:
        parts.append("[stderr]")
        parts.append(stderr_text)
    if stdout_text:
        parts.append("[stdout]")
        parts.append(stdout_text)
    if not stdout_text.strip() and not stderr_text.strip():
        parts.append("(无输出)")
    output = "\n".join(parts)
    if len(output) > MAX_FOREGROUND_DISPLAY_CHARS:
        if stderr_text:
            stderr_block = f"exit_code: {returncode}\n[stderr]\n{_tail_chars(stderr_text, MAX_FOREGROUND_DISPLAY_CHARS)}"
            if stdout_text:
                remaining = max(0, MAX_FOREGROUND_DISPLAY_CHARS - len(stderr_block) - len("\n[stdout]\n"))
                if remaining > 0:
                    stderr_block += f"\n[stdout]\n{stdout_text[:remaining]}"
            output = stderr_block
        else:
            output = build_range_preview(
                output,
                MAX_FOREGROUND_DISPLAY_CHARS,
                **extract_preview_ranges(output),
            )
    return output


def _tail_chars(text: str, char_limit: int) -> str:
    if char_limit <= 0 or len(text) <= char_limit:
        return text
    return text[-char_limit:]


def _decode_output(data: bytes, prefer_utf8: bool) -> str:
    if not data:
        return ""

    preferred = ["utf-8"] if prefer_utf8 else ["gb18030", "cp936", "utf-8"]
    local_enc = locale.getpreferredencoding(False)
    candidates = preferred + [local_enc, "utf-16-le", "utf-16", "latin-1"]

    seen: set[str] = set()
    for enc in candidates:
        name = (enc or "").lower()
        if not name or name in seen:
            continue
        seen.add(name)
        try:
            return data.decode(name)
        except UnicodeDecodeError:
            continue

    return data.decode("utf-8", errors="replace")


def _workspace_root(runtime_context: dict[str, Any] | None = None) -> Path:
    context = runtime_context or {}
    raw_cwd = context.get("cwd")
    if isinstance(raw_cwd, str) and raw_cwd.strip():
        return Path(raw_cwd).resolve()
    return Path.cwd().resolve()


def _background_dir(runtime_context: dict[str, Any] | None = None) -> Path:
    return (_workspace_root(runtime_context) / DEFAULT_BACKGROUND_DIR).resolve()


def _path_for_display(path: Path, runtime_context: dict[str, Any] | None = None) -> str:
    root = _workspace_root(runtime_context)
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return str(path.resolve())


def _resolve_path(path: str, runtime_context: dict[str, Any] | None = None) -> Path:
    root = _workspace_root(runtime_context)
    raw = Path(path)
    target = (root / raw).resolve() if not raw.is_absolute() else raw.resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("路径超出工作目录") from exc
    return target


__all__ = [
    "read_file",
    "read_many",
    "list_directory",
    "glob_files",
    "grep_search",
    "parallel_tools",
    "write_file",
    "append_file",
    "apply_diff",
    "apply_patch",
    "todo_write",
    "todo_read",
    "run_python",
    "run_powershell",
    "start_background_powershell",
    "check_background_powershell",
    "sleep_then_check_background_powershell",
    "MAX_READ_CHARS",
    "MAX_READ_FILES",
    "READ_MANY_PREVIEW_LINES",
]
