"""Shared helpers for range-style preview rendering."""

from __future__ import annotations

import re


_LINES_RE = re.compile(r"^\s*#?\s*lines:\s*(\d+)-(\d+)\s*/\s*(\d+)\s*$", re.IGNORECASE)
_PAGES_RE = re.compile(r"^\s*#?\s*pages:\s*(\d+)-(\d+)\s*/\s*(\d+)\s*$", re.IGNORECASE)


def extract_preview_ranges(text: str) -> dict[str, int]:
    """Extract line/page metadata from a structured tool result header."""
    metadata: dict[str, int] = {}
    for raw_line in text.splitlines()[:12]:
        line = raw_line.strip()
        if not line:
            continue
        match = _LINES_RE.match(line)
        if match:
            metadata["line_start"] = int(match.group(1))
            metadata["line_end"] = int(match.group(2))
            metadata["total_lines"] = int(match.group(3))
            continue
        match = _PAGES_RE.match(line)
        if match:
            metadata["page_start"] = int(match.group(1))
            metadata["page_end"] = int(match.group(2))
            metadata["total_pages"] = int(match.group(3))
    return metadata


def build_range_preview(
    text: str,
    limit: int,
    *,
    line_start: int | None = None,
    line_end: int | None = None,
    total_lines: int | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    total_pages: int | None = None,
    include_content: bool = True,
) -> str:
    """Render a range-only preview without semantic truncation wording."""
    source = text if isinstance(text, str) else str(text)
    total_chars = len(source)
    shown_chars = min(total_chars, max(0, limit))
    lines = [f"chars: {shown_chars}/{total_chars}"]

    if (
        line_start is not None
        and line_end is not None
        and total_lines is not None
    ):
        lines.append(f"lines: {line_start}-{line_end}/{total_lines}")
    if (
        page_start is not None
        and page_end is not None
        and total_pages is not None
    ):
        lines.append(f"pages: {page_start}-{page_end}/{total_pages}")

    if include_content:
        preview = source[:shown_chars].rstrip()
        if preview:
            lines.extend(["", preview])

    return "\n".join(lines).rstrip()


__all__ = ["build_range_preview", "extract_preview_ranges"]
