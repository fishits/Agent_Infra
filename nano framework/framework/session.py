"""Reusable graph session and artifact-workspace helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

from .mem_paths import MEM_ROOT as _DEFAULT_MEM_ROOT, run_dir

ArtifactCheck = tuple[str, str]
DEFAULT_MEM_ROOT = _DEFAULT_MEM_ROOT
DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


def create_run_id(prefix: str = "run") -> str:
    """Create one readable run id."""
    return f"{prefix}_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def load_env_defaults(root: str | Path | None = None) -> Path:
    """Load KEY=VALUE lines from a repo-root .env file if present."""
    resolved_root = Path(root or DEFAULT_WORKSPACE_ROOT).resolve()
    env_file = resolved_root / ".env"
    if not env_file.exists():
        return resolved_root

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
    return resolved_root


def format_run_label(run_id: str, prefix: str = "run") -> str:
    """Format run_YYYYMMDD_HHMMSS as a human readable label."""
    stamp = run_id.replace(f"{prefix}_", "", 1)
    parts = stamp.split("_")
    if len(parts) == 2 and len(parts[0]) == 8 and len(parts[1]) == 6:
        d, t = parts
        return f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
    return run_id


def node_memory_hint(run_id: str, mem_root: Path) -> str:
    """Return a compact list of node memories persisted for a run."""
    mem_dir = run_dir(run_id, mem_root)
    if not mem_dir.is_dir():
        return ""
    node_files = [file.stem for file in mem_dir.glob("*.json") if not file.stem.endswith(".state")]
    return ", ".join(sorted(node_files))


def trace_count(workdir: Path, trace_name: str = "trace.json") -> int:
    """Return number of trace entries for a workspace."""
    trace_path = workdir / trace_name
    loaded = read_json(trace_path, default=[])
    return len(loaded) if isinstance(loaded, list) else 0


def list_existing_runs(
    *,
    mem_root: Path | None = None,
    workspace_root: Path | None = None,
    run_prefix: str = "run",
    label_builder: Callable[[str], str] | None = None,
) -> list[tuple[str, str]]:
    """Scan memory/workspace roots for existing run folders."""
    resolved_mem_root = Path(mem_root or DEFAULT_MEM_ROOT)
    run_ids: set[str] = set()
    for root in (resolved_mem_root, workspace_root):
        if root is not None and root.is_dir():
            run_ids.update(child.name for child in root.iterdir() if child.is_dir() and child.name.startswith(f"{run_prefix}_"))

    runs = []
    for run_id in sorted(run_ids):
        label = label_builder(run_id) if label_builder else format_run_label(run_id, run_prefix)
        hint = node_memory_hint(run_id, resolved_mem_root)
        if hint:
            label += f"  [{hint}]"
        runs.append((run_id, label))
    return runs


def pick_run_id(
    *,
    title: str = "Previous sessions",
    mem_root: Path | None = None,
    workspace_root: Path | None = None,
    run_prefix: str = "run",
    label_builder: Callable[[str], str] | None = None,
) -> str | None:
    """Interactive run picker. Returns an existing run_id or None for a new session."""
    runs = list_existing_runs(
        mem_root=mem_root,
        workspace_root=workspace_root,
        run_prefix=run_prefix,
        label_builder=label_builder,
    )
    if not runs:
        print("No previous sessions found. Starting a new session.\n")
        return None

    print(f"{title}:")
    print("-" * 60)
    for index, (run_id, label) in enumerate(runs, start=1):
        marker = " (latest)" if index == len(runs) else ""
        print(f"  [{index}] {label}{marker}")
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


def read_json(path: Path, default: Any = None) -> Any:
    """Read JSON with a default for missing or invalid files."""
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json(path: Path, payload: Any) -> None:
    """Write JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def init_run_workspace(
    *,
    run_id: str,
    workspace_root: Path,
    subdirs: Iterable[str] = (),
    meta: dict[str, Any] | None = None,
    trace_name: str = "trace.json",
    meta_name: str = "run_meta.json",
) -> Path:
    """Create a run workspace with optional subdirs, trace, and metadata."""
    workdir = workspace_root / run_id
    workdir.mkdir(parents=True, exist_ok=True)
    for subdir in subdirs:
        (workdir / subdir).mkdir(parents=True, exist_ok=True)

    trace_path = workdir / trace_name
    if not trace_path.exists():
        trace_path.write_text("[]", encoding="utf-8")

    meta_path = workdir / meta_name
    if meta is not None and not meta_path.exists():
        write_json(meta_path, meta)
    return workdir


def first_missing_artifact(round_dir: Path, checks: list[ArtifactCheck]) -> str | None:
    """Return the node for the first missing artifact in a round."""
    for file_name, node_name in checks:
        if not (round_dir / file_name).exists():
            return node_name
    return None


def infer_round_state(
    *,
    workdir: Path,
    artifact_checks: list[ArtifactCheck],
    start_node: str,
    round_prefix: str = "round",
    round_width: int = 3,
    extra_state: Callable[[Path], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Infer current round id/index/current node from artifact files."""
    rounds_dir = workdir / "rounds"
    rounds = sorted(path for path in rounds_dir.glob(f"{round_prefix}_*") if path.is_dir())
    extra = extra_state(workdir) if extra_state else {}

    for round_dir in rounds:
        missing_node = first_missing_artifact(round_dir, artifact_checks)
        if missing_node is not None:
            round_index = round_index_from_id(round_dir.name)
            return {
                "round_id": round_dir.name,
                "round_index": round_index,
                "current_node": missing_node,
                **extra,
            }

    next_round_index = len(rounds) + 1
    return {
        "round_id": f"{round_prefix}_{next_round_index:0{round_width}d}",
        "round_index": next_round_index,
        "current_node": start_node,
        **extra,
    }


def round_index_from_id(round_id: str) -> int:
    """Parse round_001 style ids."""
    try:
        return int(round_id.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 1


def infer_sota_round_id(
    workdir: Path,
    *,
    trace_name: str = "trace.json",
    decision_key: str = "decision",
    round_key: str = "round_id",
) -> str | None:
    """Infer latest accepted round id from a lightweight trace."""
    trace = read_json(workdir / trace_name, default=[])
    if not isinstance(trace, list):
        return None
    for item in reversed(trace):
        if isinstance(item, dict) and item.get(decision_key) is True:
            value = item.get(round_key)
            return str(value) if value else None
    return None


__all__ = [
    "ArtifactCheck",
    "DEFAULT_MEM_ROOT",
    "DEFAULT_WORKSPACE_ROOT",
    "create_run_id",
    "first_missing_artifact",
    "format_run_label",
    "infer_round_state",
    "infer_sota_round_id",
    "init_run_workspace",
    "list_existing_runs",
    "load_env_defaults",
    "node_memory_hint",
    "pick_run_id",
    "read_json",
    "round_index_from_id",
    "trace_count",
    "write_json",
]
