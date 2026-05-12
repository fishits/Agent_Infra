from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from framework.context.store import read_records
from framework.mem_paths import MEM_ROOT, run_dir


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = str((REPO_ROOT.parent / "input").resolve())
BOSS_SCRIPT = REPO_ROOT / "OnlyTeam" / "boss.py"
NEW_SESSION_KEY = "__new_session__"
LLM_METRIC_RE = re.compile(
    r"\[LLM\]\s+TTFT=(?P<ttft>[0-9.]+)s\s+time=(?P<time>[0-9.]+)s\s+decode=(?P<decode>[0-9.]+)\s+tok/s\s+in=(?P<input>\S+)\s+out=(?P<output>\S+)\s+cache=(?P<cache>\S+)"
)


def _ensure_state() -> None:
    st.session_state.setdefault("ui_runs", {})
    st.session_state.setdefault("active_run_key", NEW_SESSION_KEY)
    st.session_state.setdefault("active_view", "Boss")
    st.session_state.setdefault("trajectory_mode", False)
    st.session_state.setdefault("workspace_text", DEFAULT_WORKSPACE)


def _make_run_id() -> str:
    return f"onlyteam_boss_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _ui_runs_dir(workspace: Path) -> Path:
    target = workspace / "coding" / "ui_runs"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _default_workspace_for_run(run_id: str) -> str:
    workspace = Path(DEFAULT_WORKSPACE)
    dispatch_path = workspace / "coding" / "dispatches" / run_id
    if dispatch_path.exists():
        return str(workspace)
    return DEFAULT_WORKSPACE


def _history_run_info(run_id: str) -> dict[str, Any]:
    return {
        "run_key": run_id,
        "run_id": run_id,
        "mode": "Boss",
        "workspace": _default_workspace_for_run(run_id),
        "pid": "",
        "started_at": "",
        "log_path": "",
        "process": None,
        "stdin_handle": None,
        "stream_handle": None,
    }


def _load_sessions_from_mem() -> None:
    runs = st.session_state["ui_runs"]
    if not MEM_ROOT.exists():
        return
    for child in sorted(MEM_ROOT.iterdir()):
        if not child.is_dir():
            continue
        run_id = child.name
        if not run_id.startswith("onlyteam_boss_"):
            continue
        if run_id not in runs:
            runs[run_id] = _history_run_info(run_id)


def _open_log_stream(workspace: Path, run_id: str) -> tuple[Path, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = _ui_runs_dir(workspace) / f"boss_{timestamp}_{run_id}.log"
    stream = log_path.open("a", encoding="utf-8", errors="replace")
    stream.write(f"[ui] started_at={datetime.now().isoformat()}\n")
    stream.flush()
    return log_path, stream


def _start_boss_process(workspace: Path, run_id: str, first_message: str) -> dict[str, Any]:
    log_path, stream = _open_log_stream(workspace, run_id)
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        [
            sys.executable,
            "-u",
            str(BOSS_SCRIPT),
            "--stream",
            "--run-id",
            run_id,
        ],
        cwd=workspace,
        stdin=subprocess.PIPE,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=creationflags,
    )
    assert process.stdin is not None
    process.stdin.write(first_message.strip() + "\n")
    process.stdin.flush()
    return {
        "run_key": run_id,
        "run_id": run_id,
        "mode": "Boss",
        "workspace": str(workspace),
        "pid": process.pid,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "log_path": str(log_path),
        "process": process,
        "stdin_handle": process.stdin,
        "stream_handle": stream,
    }


def _launch_new_boss(workspace: Path, first_message: str) -> tuple[str, dict[str, Any]]:
    run_id = _make_run_id()
    return run_id, _start_boss_process(workspace, run_id, first_message)


def _resume_boss(run_info: dict[str, Any], first_message: str) -> dict[str, Any]:
    workspace = Path(str(run_info["workspace"])).expanduser().resolve()
    return _start_boss_process(workspace, str(run_info["run_id"]), first_message)


def _get_active_run() -> dict[str, Any] | None:
    key = st.session_state.get("active_run_key")
    if not key or key == NEW_SESSION_KEY:
        return None
    return st.session_state["ui_runs"].get(key)


def _current_status(run_info: dict[str, Any] | None) -> str:
    if not run_info:
        return "new"
    process = run_info.get("process")
    if process is None:
        return "history"
    return "running" if process.poll() is None else f"exited ({process.poll()})"


def _status_badge(status: str) -> str:
    if status == "new":
        return "新会话"
    if status.startswith("running"):
        return "进行中"
    if status.startswith("exited (0)"):
        return "已完成"
    if status.startswith("exited"):
        return "失败"
    if status == "history":
        return "历史"
    return "未知"


def _maybe_close_handles(run_info: dict[str, Any] | None) -> None:
    if not run_info:
        return
    process = run_info.get("process")
    stream = run_info.get("stream_handle")
    stdin_handle = run_info.get("stdin_handle")
    if process is not None and process.poll() is not None:
        if stream is not None and not stream.closed:
            stream.close()
        if stdin_handle is not None and not stdin_handle.closed:
            stdin_handle.close()


def _send_to_running_boss(run_info: dict[str, Any], message: str) -> str | None:
    process = run_info.get("process")
    stdin_handle = run_info.get("stdin_handle")
    if process is None or process.poll() is not None:
        return "当前 Boss 会话没有在运行。"
    if stdin_handle is None or stdin_handle.closed:
        return "当前 Boss 会话的输入通道已关闭。"
    payload = message.strip()
    if not payload:
        return "输入不能为空。"
    stdin_handle.write(payload + "\n")
    stdin_handle.flush()
    return None


def _terminate_run(run_info: dict[str, Any] | None) -> None:
    if not run_info:
        return
    process = run_info.get("process")
    stdin_handle = run_info.get("stdin_handle")
    stream = run_info.get("stream_handle")
    if stdin_handle is not None and not stdin_handle.closed:
        stdin_handle.close()
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=3)
    if stream is not None and not stream.closed:
        stream.close()


def _safe_rmtree(target: Path, anchor: Path) -> None:
    resolved_target = target.resolve()
    resolved_anchor = anchor.resolve()
    try:
        resolved_target.relative_to(resolved_anchor)
    except ValueError:
        return
    if resolved_target.exists():
        shutil.rmtree(resolved_target, ignore_errors=True)


def _safe_unlink(target: Path, anchor: Path) -> None:
    resolved_target = target.resolve()
    resolved_anchor = anchor.resolve()
    try:
        resolved_target.relative_to(resolved_anchor)
    except ValueError:
        return
    if resolved_target.exists() and resolved_target.is_file():
        resolved_target.unlink(missing_ok=True)


def _delete_session(run_key: str) -> None:
    runs = st.session_state["ui_runs"]
    run_info = runs.pop(run_key, None)
    if run_info is None:
        return

    _terminate_run(run_info)

    run_id = str(run_info["run_id"])
    _safe_rmtree(run_dir(run_id, MEM_ROOT), MEM_ROOT)

    workspace = Path(str(run_info["workspace"])).expanduser().resolve()
    _safe_rmtree(workspace / "coding" / "dispatches" / run_id, workspace / "coding" / "dispatches")
    log_path = str(run_info.get("log_path") or "").strip()
    if log_path:
        _safe_unlink(Path(log_path), workspace / "coding" / "ui_runs")

    if st.session_state.get("active_run_key") == run_key:
        st.session_state["active_run_key"] = NEW_SESSION_KEY
        st.session_state["active_view"] = "Boss"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _get_dispatch_summary(workspace: Path, run_id: str) -> dict[str, Any] | None:
    dispatch_path = workspace / "coding" / "dispatches" / run_id / "boss_dispatch.json"
    return _load_json(dispatch_path)


def _get_subagent_runs(run_info: dict[str, Any]) -> list[dict[str, Any]]:
    workspace = Path(str(run_info["workspace"])).expanduser().resolve()
    summary = _get_dispatch_summary(workspace, str(run_info["run_id"]))
    if summary is None:
        return []
    return [item for item in summary.get("subagent_runs", []) if isinstance(item, dict)]


def _session_title(run_info: dict[str, Any]) -> str:
    records = read_records(str(run_info["run_id"]), "onlyteam_boss")
    for record in reversed(records):
        if record.get("role") == "user":
            content = str(record.get("content") or "").strip()
            if content:
                return _trim_text(content, 26)
    return str(run_info["run_id"])


def _session_meta(run_info: dict[str, Any]) -> str:
    run_id = str(run_info["run_id"])
    created = _run_id_to_time(run_id)
    status = _status_badge(_current_status(run_info))
    return f"{created} · {status}"


def _run_id_to_time(run_id: str) -> str:
    parts = run_id.split("_")
    if len(parts) >= 4:
        day = parts[-2]
        clock = parts[-1]
        if len(day) == 8 and len(clock) == 6:
            return f"{day[4:6]}-{day[6:8]} {clock[:2]}:{clock[2:4]}"
    return run_id


def _trim_text(text: str, limit: int) -> str:
    value = " ".join(text.split())
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _view_options(run_info: dict[str, Any]) -> list[str]:
    options = ["Boss"]
    sub_runs = _get_subagent_runs(run_info)
    options.extend([f"Sub-Agent {index}" for index in range(1, len(sub_runs) + 1)])
    return options


def _render_session_sidebar() -> None:
    st.sidebar.title("Sessions")
    _load_sessions_from_mem()

    if st.sidebar.button("+ 新会话", use_container_width=True):
        st.session_state["active_run_key"] = NEW_SESSION_KEY
        st.session_state["active_view"] = "Boss"
        st.rerun()

    st.sidebar.text_input("Workspace", key="workspace_text")
    st.sidebar.caption("只给 Boss 发消息。选中旧会话后继续输入即可续聊。")
    st.sidebar.divider()

    runs = st.session_state["ui_runs"]
    if not runs:
        st.sidebar.caption("还没有会话。")
        return

    active_key = st.session_state.get("active_run_key")
    for run_key in sorted(runs.keys(), reverse=True):
        run_info = runs[run_key]
        row = st.sidebar.columns([5, 1])
        if row[0].button(
            _session_title(run_info),
            key=f"session_{run_key}",
            use_container_width=True,
            type="primary" if run_key == active_key else "secondary",
        ):
            st.session_state["active_run_key"] = run_key
            st.session_state["active_view"] = "Boss"
            st.rerun()
        if row[1].button("删", key=f"delete_{run_key}", use_container_width=True):
            _delete_session(run_key)
            st.rerun()
        st.sidebar.caption(_session_meta(run_info))


def _inject_styles() -> None:
    st.markdown(
        """
<style>
div[data-testid="stChatMessage"] {
  border: 1px solid #d9dde7;
  border-radius: 14px;
  padding: 0.55rem 0.8rem;
  margin-bottom: 0.6rem;
  background: #ffffff;
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
  margin-bottom: 0.2rem;
}
div[data-testid="stExpander"] details {
  border: 1px solid #d9dde7;
  border-radius: 14px;
  background: #ffffff;
}
div[data-testid="stExpander"] details summary p {
  font-weight: 600;
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _resolve_log_path(run_info: dict[str, Any]) -> Path | None:
    raw = str(run_info.get("log_path") or "").strip()
    if raw:
        path = Path(raw)
        if path.exists():
            return path

    workspace = Path(str(run_info["workspace"])).expanduser().resolve()
    ui_runs_dir = workspace / "coding" / "ui_runs"
    if not ui_runs_dir.exists():
        return None

    run_id = str(run_info["run_id"])
    files = sorted(ui_runs_dir.glob("*.log"), key=lambda item: item.stat().st_mtime, reverse=True)

    for path in files:
        if run_id in path.name:
            return path

    for path in files[:20]:
        try:
            tail = path.read_text(encoding="utf-8", errors="replace")[-4000:]
        except OSError:
            continue
        if run_id in tail:
            return path
    return None


def _load_llm_metrics(run_info: dict[str, Any]) -> list[dict[str, str]]:
    log_path = _resolve_log_path(run_info)
    if log_path is None or not log_path.exists():
        return []

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    metrics: list[dict[str, str]] = []
    for line in text.splitlines():
        match = LLM_METRIC_RE.search(line)
        if not match:
            continue
        metrics.append(
            {
                "ttft": match.group("ttft"),
                "time": match.group("time"),
                "decode": match.group("decode"),
                "input": match.group("input"),
                "output": match.group("output"),
                "cache": match.group("cache"),
            }
        )
    return metrics


def _render_text_bubble(body: str, *, label: str | None = None) -> None:
    with st.chat_message("assistant"):
        if label:
            st.markdown(f"**{label}**")
        st.text(body)


def _render_markdown_bubble(body: str, *, label: str | None = None) -> None:
    with st.chat_message("assistant"):
        if label:
            st.markdown(f"**{label}**")
        st.markdown(body)


def _render_function_call_bubble(record: dict[str, Any]) -> None:
    name = str(record.get("name") or "").strip() or "unknown_tool"
    arguments = str(record.get("arguments") or "").strip()
    thinking = str(record.get("thinking") or "").strip()
    with st.chat_message("assistant"):
        st.markdown(f"**Tool Call** `{name}`")
        if thinking:
            st.markdown("**Thinking**")
            st.text(thinking)
        if arguments:
            st.markdown("**Arguments**")
            st.code(arguments, language="json")


def _render_function_output_bubble(record: dict[str, Any]) -> None:
    output = str(record.get("output") or "").strip()
    if not output:
        return
    if output.startswith("CHAT:"):
        _render_markdown_bubble(output[5:].strip())
        return
    if output.startswith("DECIDE:end:"):
        _render_markdown_bubble(output[len("DECIDE:end:") :].strip(), label="Final")
        return
    _render_text_bubble(output, label="Tool Result")


def _render_reasoning_bubble(record: dict[str, Any]) -> None:
    summary = record.get("summary")
    if isinstance(summary, list):
        text = "\n".join(str(item) for item in summary if item)
    else:
        text = str(summary or "")
    text = text.strip()
    if text:
        _render_text_bubble(text, label="Reasoning Summary")


def _compact_metric_line(record: dict[str, Any], metric: dict[str, str] | None) -> str:
    parts: list[str] = []
    ttft = record.get("ttft")
    total_time = record.get("total_time")
    decode_tokens_per_second = record.get("decode_tokens_per_second")
    cached_tokens = record.get("cached_tokens")
    cache_text = str(record.get("cache_text") or "").strip()

    if isinstance(ttft, (int, float)):
        parts.append(f"ttft {ttft:.3f}s")
    elif metric is not None:
        ttft_text = str(metric.get("ttft") or "").strip()
        if ttft_text:
            parts.append(f"ttft {ttft_text}s")

    if isinstance(total_time, (int, float)):
        parts.append(f"total {total_time:.3f}s")
    elif metric is not None:
        total_text = str(metric.get("time") or "").strip()
        if total_text:
            parts.append(f"total {total_text}s")

    if isinstance(decode_tokens_per_second, (int, float)):
        parts.append(f"decode {decode_tokens_per_second:.2f} tok/s")
    elif metric is not None:
        decode_text = str(metric.get("decode") or "").strip()
        if decode_text:
            parts.append(f"decode {decode_text} tok/s")

    if cache_text:
        parts.append(f"cache {cache_text}")
    elif isinstance(cached_tokens, int):
        parts.append(f"cache {cached_tokens}")
    elif metric is not None:
        cache = str(metric.get("cache") or "").strip()
        if cache and cache.lower() != "n/a":
            parts.append(f"cache {cache}")

    input_tokens = record.get("input_tokens")
    output_tokens = record.get("output_tokens")
    if isinstance(input_tokens, int):
        parts.append(f"input {input_tokens}")
    if isinstance(output_tokens, int):
        parts.append(f"output {output_tokens}")
    return " · ".join(parts)


def _render_token_usage_bubble(record: dict[str, Any], metric: dict[str, str] | None) -> None:
    line = _compact_metric_line(record, metric)
    if not line:
        return
    with st.chat_message("assistant"):
        st.markdown("**LLM Metrics**")
        st.caption(line)

        created_at = str(record.get("created_at") or "").strip()
        response_id = str(record.get("response_id") or "").strip()
        meta_parts = [part for part in [created_at, response_id] if part]
        if meta_parts:
            st.caption(" / ".join(meta_parts))


def _preview_text(text: str, limit: int = 320) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _collect_thinking(items: list[dict[str, Any]]) -> str:
    seen: set[str] = set()
    chunks: list[str] = []
    for item in items:
        if item.get("type") == "reasoning":
            summary = item.get("summary")
            if isinstance(summary, list):
                text = "\n".join(str(part) for part in summary if part).strip()
            else:
                text = str(summary or "").strip()
        else:
            text = str(item.get("thinking") or "").strip()
        if text and text not in seen:
            seen.add(text)
            chunks.append(text)
    return "\n\n".join(chunks)


def _build_trajectory_rounds(records: list[dict[str, Any]], metrics: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    pending_user_messages: list[str] = []
    active: dict[str, Any] | None = None
    metric_index = 0
    metrics = metrics or []

    def start_round() -> dict[str, Any]:
        nonlocal pending_user_messages
        round_data = {
            "user_messages": pending_user_messages,
            "response_items": [],
            "outputs": [],
            "usage": None,
            "metric": None,
        }
        pending_user_messages = []
        return round_data

    def flush_round() -> None:
        nonlocal active
        if active is None:
            return
        if active["user_messages"] or active["response_items"] or active["outputs"] or active["usage"] is not None:
            rounds.append(active)
        active = None

    for record in records:
        role = record.get("role")
        item_type = record.get("type")

        if role == "user":
            content = str(record.get("content") or "").strip()
            if active is not None and (active["response_items"] or active["outputs"] or active["usage"] is not None):
                flush_round()
            if content:
                pending_user_messages.append(content)
            continue

        if role == "assistant" or item_type in {"reasoning", "function_call"}:
            if active is None:
                active = start_round()
            elif active["usage"] is not None and active["outputs"]:
                flush_round()
                active = start_round()
            active["response_items"].append(record)
            continue

        if item_type == "token_usage":
            if active is None:
                active = start_round()
            active["usage"] = record
            active["metric"] = metrics[metric_index] if metric_index < len(metrics) else None
            metric_index += 1
            continue

        if item_type == "function_call_output":
            if active is None:
                active = start_round()
            active["outputs"].append(record)
            continue

    flush_round()
    return rounds


def _render_trajectory_view(records: list[dict[str, Any]], metrics: list[dict[str, str]] | None = None) -> None:
    rounds = _build_trajectory_rounds(records, metrics)
    if not rounds:
        st.info("还没有可展示的轨迹。")
        return

    for index, round_data in enumerate(rounds, start=1):
        response_items = round_data["response_items"]
        outputs = round_data["outputs"]
        usage = round_data["usage"]
        metric = round_data["metric"]
        thinking = _collect_thinking(response_items)
        tool_names = [
            str(item.get("name") or "").strip()
            for item in response_items
            if item.get("type") == "function_call"
        ]
        tool_names = [name for name in tool_names if name]

        title_parts = [f"Round {index}"]
        if tool_names:
            title_parts.append(" → ".join(tool_names[:3]) + (" +…" if len(tool_names) > 3 else ""))
        if usage is not None:
            metric_line = _compact_metric_line(usage, metric)
            if metric_line:
                title_parts.append(metric_line)

        with st.expander(" | ".join(title_parts), expanded=False):
            if round_data["user_messages"]:
                st.markdown("**User Input**")
                for message in round_data["user_messages"]:
                    st.markdown(message)

            if thinking:
                st.markdown("**Thinking**")
                st.text(thinking)

            if tool_names:
                st.markdown("**Tool Sequence**")
                for item in response_items:
                    if item.get("type") != "function_call":
                        continue
                    name = str(item.get("name") or "").strip() or "unknown_tool"
                    st.markdown(f"- `{name}`")
                    arguments = str(item.get("arguments") or "").strip()
                    if arguments:
                        st.code(arguments, language="json")

            if outputs:
                st.markdown("**Tool Results**")
                for output in outputs:
                    text = str(output.get("output") or "").strip()
                    if text:
                        st.caption(_preview_text(text, 700))

            if usage is not None:
                line = _compact_metric_line(usage, metric)
                if line:
                    st.markdown("**LLM Metrics**")
                    st.caption(line)


def _render_record_list(records: list[dict[str, Any]], metrics: list[dict[str, str]] | None = None) -> None:
    if not records:
        st.info("还没有记录。")
        return

    metrics = metrics or []
    metric_index = 0

    for record in records:
        role = record.get("role")
        item_type = record.get("type")

        if role == "user":
            content = str(record.get("content") or "").strip()
            if content:
                with st.chat_message("user"):
                    st.markdown(content)
            continue

        if role == "assistant":
            content = str(record.get("content") or "").strip()
            thinking = str(record.get("thinking") or "").strip()
            if content or thinking:
                with st.chat_message("assistant"):
                    if content:
                        st.markdown(content)
                    if thinking:
                        st.markdown("**Thinking**")
                        st.text(thinking)
            continue

        if item_type == "reasoning":
            _render_reasoning_bubble(record)
            continue

        if item_type == "function_call":
            _render_function_call_bubble(record)
            continue

        if item_type == "function_call_output":
            _render_function_output_bubble(record)
            continue

        if item_type == "token_usage":
            metric = metrics[metric_index] if metric_index < len(metrics) else None
            _render_token_usage_bubble(record, metric)
            metric_index += 1


def _render_boss_view(run_info: dict[str, Any]) -> None:
    records = read_records(str(run_info["run_id"]), "onlyteam_boss")
    metrics = _load_llm_metrics(run_info)
    if st.session_state.get("trajectory_mode"):
        _render_trajectory_view(records, metrics)
        return
    _render_record_list(records, metrics)


def _render_subagent_view(launch: dict[str, Any], index: int) -> None:
    bias = str(launch.get("weak_bias") or "").strip()
    if bias:
        st.caption(f"Sub-Agent {index} · {bias}")
    run_id = str(launch.get("run_id") or "")
    records = read_records(run_id, "onlyteam_subagent")
    if st.session_state.get("trajectory_mode"):
        _render_trajectory_view(records)
        return
    _render_record_list(records)


def _render_view_buttons(run_info: dict[str, Any]) -> None:
    options = _view_options(run_info)
    current_view = st.session_state.get("active_view", "Boss")
    if current_view not in options:
        current_view = "Boss"
        st.session_state["active_view"] = current_view

    columns = st.columns(len(options) + 1)
    for index, option in enumerate(options):
        if columns[index].button(
            option,
            key=f"view_{run_info['run_id']}_{option}",
            use_container_width=True,
            type="primary" if option == current_view else "secondary",
        ):
            st.session_state["active_view"] = option
            st.rerun()
    trajectory_mode = bool(st.session_state.get("trajectory_mode"))
    if columns[-1].button(
        "轨迹模式 开" if trajectory_mode else "轨迹模式 关",
        key=f"trajectory_{run_info['run_id']}",
        use_container_width=True,
        type="primary" if trajectory_mode else "secondary",
    ):
        st.session_state["trajectory_mode"] = not trajectory_mode
        st.rerun()


@st.fragment(run_every="2s")
def _live_panel() -> None:
    run_info = _get_active_run()
    if not run_info:
        st.info("新会话，直接在下面输入发给 Boss。")
        return

    _maybe_close_handles(run_info)
    status = _status_badge(_current_status(run_info))
    st.caption(f"{status} · {run_info['run_id']}")
    _render_view_buttons(run_info)

    current_view = st.session_state.get("active_view", "Boss")
    if current_view == "Boss":
        _render_boss_view(run_info)
        return

    sub_runs = _get_subagent_runs(run_info)
    index = int(current_view.replace("Sub-Agent ", "")) - 1
    if 0 <= index < len(sub_runs):
        _render_subagent_view(sub_runs[index], index + 1)
    else:
        st.info("这个 Sub-Agent 还不存在。")


def _submit_message(message: str) -> str | None:
    payload = message.strip()
    if not payload:
        return "输入不能为空。"

    active_run = _get_active_run()
    if active_run is None:
        workspace = Path(str(st.session_state["workspace_text"])).expanduser().resolve()
        if not workspace.exists():
            return f"workspace 不存在：{workspace}"
        run_key, run_info = _launch_new_boss(workspace, payload)
        st.session_state["ui_runs"][run_key] = run_info
        st.session_state["active_run_key"] = run_key
        st.session_state["active_view"] = "Boss"
        return None

    _maybe_close_handles(active_run)
    process = active_run.get("process")
    if process is not None and process.poll() is None:
        return _send_to_running_boss(active_run, payload)

    resumed = _resume_boss(active_run, payload)
    st.session_state["ui_runs"][str(active_run["run_key"])] = resumed
    st.session_state["active_run_key"] = str(active_run["run_key"])
    st.session_state["active_view"] = "Boss"
    return None


def main() -> None:
    st.set_page_config(page_title="OnlyTeam Boss", layout="wide")
    _ensure_state()
    _inject_styles()
    _render_session_sidebar()

    st.title("OnlyTeam Boss")
    _live_panel()

    prompt = st.chat_input("给 Boss 发消息")
    if prompt is not None:
        error = _submit_message(prompt)
        if error:
            st.error(error)
        else:
            st.rerun()


if __name__ == "__main__":
    main()
