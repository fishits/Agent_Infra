"""Helpers for mapping run ids to on-disk .mem directories."""

from __future__ import annotations

import re
from pathlib import Path


MEM_ROOT = Path(__file__).resolve().parent.parent / ".mem"
_BOSS_SUBAGENT_RE = re.compile(r"^(onlyteam_boss_\d{8}_\d{6})_subagent_(\d+)$")


def run_dir(run_id: str, mem_root: Path | None = None) -> Path:
    root = Path(mem_root or MEM_ROOT)
    match = _BOSS_SUBAGENT_RE.match(str(run_id))
    if not match:
        return root / str(run_id)
    boss_run_id, subagent_index = match.groups()
    return root / boss_run_id / "subagents" / f"subagent_{subagent_index}"


__all__ = ["MEM_ROOT", "run_dir"]
