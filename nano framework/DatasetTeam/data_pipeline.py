"""DatasetTeam RL preference extraction and export helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from framework.llm import openai_llm


SEVERE_FAILURE_MODES = {
    "fabricated_evidence",
    "unsupported_claim",
    "unsafe_tool_use",
    "phase_violation",
    "invalid_format",
    "no_task_progress",
}


def list_dataset_runs(workspace: str | Path | None = None) -> dict[str, Any]:
    root = Path(workspace or Path.cwd()).resolve()
    mem_root = root / ".mem"
    runs: list[dict[str, Any]] = []
    if not mem_root.exists():
        return {"mem_root": str(mem_root), "runs": []}
    for run_dir in sorted(mem_root.iterdir()):
        if not run_dir.is_dir():
            continue
        history = run_dir / "only_team.json"
        if not history.exists():
            continue
        count = 0
        try:
            loaded = json.loads(history.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                count = len(loaded)
        except json.JSONDecodeError:
            count = 0
        runs.append(
            {
                "source_run_id": run_dir.name,
                "history_path": str(history),
                "record_count": count,
            }
        )
    return {"mem_root": str(mem_root), "runs": runs}


def read_mem_run(workspace: str | Path | None, source_run_id: str) -> dict[str, Any]:
    path = _source_mem_path(workspace, source_run_id)
    records = _load_json_list(path)
    type_counts: dict[str, int] = {}
    for record in records:
        key = str(record.get("type") or record.get("role") or "unknown")
        type_counts[key] = type_counts.get(key, 0) + 1
    return {
        "source_run_id": source_run_id,
        "history_path": str(path),
        "exists": path.exists(),
        "record_count": len(records),
        "type_counts": type_counts,
        "preview_user_message": _first_nonempty_user_message(records),
    }


def build_decision_records_file(
    *,
    workspace: str | Path | None,
    source_run_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    history_path = _source_mem_path(workspace, source_run_id)
    records = _load_json_list(history_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "decision_records.jsonl"

    goal_hint = _first_nonempty_user_message(records)
    built: list[dict[str, Any]] = []
    missing_outputs = 0

    for index, record in enumerate(records):
        if record.get("type") != "function_call":
            continue
        call_id = str(record.get("call_id") or "").strip()
        tool_name = str(record.get("name") or "").strip()
        if not call_id or not tool_name:
            continue
        output_record, output_index = _find_output_record(records, call_id, index + 1)
        if output_record is None:
            missing_outputs += 1
            continue
        response_id = _nearest_response_id(records, index, output_index)
        recent_user_message = _recent_user_message(records, index)
        tool_arguments = str(record.get("arguments") or "")
        built.append(
            {
                "schema_version": "decision_record.v1",
                "record_id": f"{source_run_id}:{call_id}",
                "source": {
                    "source_run_id": source_run_id,
                    "node_name": "only_team",
                    "response_id": response_id,
                    "call_id": call_id,
                    "record_index": index,
                },
                "state": {
                    "recent_user_message": recent_user_message,
                    "phase": None,
                    "goal_hint": goal_hint,
                    "context_snippet": _build_context_snippet(recent_user_message, tool_name),
                },
                "thinking": {
                    "raw": str(record.get("thinking") or ""),
                    "clean": _clean_text(str(record.get("thinking") or "")),
                },
                "decision": {
                    "tool_name": tool_name,
                    "tool_arguments": tool_arguments,
                    "tool_arguments_json": _parse_json_object(tool_arguments),
                },
                "observation": {
                    "tool_output": str(output_record.get("output") or ""),
                    "status": "observed",
                },
            }
        )

    _write_jsonl(output_path, built)
    manifest = {
        "source_run_id": source_run_id,
        "history_path": str(history_path),
        "decision_records_path": str(output_path),
        "decision_records_total": len(built),
        "missing_outputs_skipped": missing_outputs,
    }
    _write_json(output_root / "decision_records_manifest.json", manifest)
    return manifest


def judge_decision_records_file(
    *,
    decision_records_path: str | Path,
    output_dir: str | Path,
    judge_model: str = "deepseekv4-pro",
    judge_reasoning_effort: str = "medium",
    judge_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    decision_path = Path(decision_records_path)
    records = _read_jsonl(decision_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    judge_path = output_root / "judge_records.jsonl"

    results: list[dict[str, Any]] = []
    accepted = 0
    judge_errors = 0
    exportable = 0

    evaluator = judge_fn or (lambda record: _judge_record_with_openai(record, judge_model, judge_reasoning_effort))
    for record in records:
        canonical_prompt = _build_training_prompt(record)
        canonical_chosen = _build_canonical_chosen(record)
        try:
            payload = evaluator(record)
            normalized = normalize_judge_payload(payload, canonical_chosen)
            valid = True
            error = ""
        except Exception as exc:
            normalized = _invalid_judge_payload(str(exc), canonical_chosen)
            valid = False
            error = str(exc)
        label = normalized["judge"]["label"]
        if label == "accept":
            accepted += 1
        if normalized["export_decision"]["openrlhf_preference"]:
            exportable += 1
        if not valid:
            judge_errors += 1

        results.append(
            {
                "schema_version": "judge_record.v1",
                "record_id": record.get("record_id"),
                "source": record.get("source", {}),
                "prompt": canonical_prompt,
                "chosen_response": normalized["chosen_response"],
                "judge": normalized["judge"],
                "rejected_response": normalized["rejected_response"],
                "export_decision": normalized["export_decision"],
                "judge_error": error,
            }
        )

    _write_jsonl(judge_path, results)
    manifest = {
        "decision_records_path": str(decision_path),
        "judge_records_path": str(judge_path),
        "judge_model": judge_model,
        "judge_reasoning_effort": judge_reasoning_effort,
        "judged_total": len(results),
        "accepted_total": accepted,
        "judge_errors": judge_errors,
        "exportable_preference_total": exportable,
    }
    _write_json(output_root / "judge_manifest.json", manifest)
    return manifest


def export_openrlhf_preference_file(
    *,
    judge_records_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    judge_path = Path(judge_records_path)
    records = _read_jsonl(judge_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    strict_path = output_root / "openrlhf_preference.strict.jsonl"
    rich_path = output_root / "openrlhf_preference.rich.jsonl"
    manifest_path = output_root / "export_manifest.json"

    strict_rows: list[dict[str, str]] = []
    rich_rows: list[dict[str, Any]] = []
    skipped_non_exportable = 0
    skipped_invalid = 0

    for record in records:
        if not bool((record.get("export_decision") or {}).get("openrlhf_preference")):
            skipped_non_exportable += 1
            continue
        prompt = str(record.get("prompt") or "").strip()
        chosen_response = str(record.get("chosen_response") or "").strip()
        rejected_response = str(record.get("rejected_response") or "").strip()
        if not prompt or not chosen_response or not rejected_response:
            skipped_invalid += 1
            continue
        strict_row = {
            "chosen": f"User: {prompt}\nAssistant: {chosen_response}",
            "rejected": f"User: {prompt}\nAssistant: {rejected_response}",
        }
        strict_rows.append(strict_row)
        judge = record.get("judge") or {}
        source = record.get("source") or {}
        rich_rows.append(
            {
                "prompt": prompt,
                "chosen": chosen_response,
                "rejected": rejected_response,
                "score": judge.get("score"),
                "confidence": judge.get("confidence"),
                "source_run_id": source.get("source_run_id"),
                "call_id": source.get("call_id"),
                "record_id": record.get("record_id"),
            }
        )

    _write_jsonl(strict_path, strict_rows)
    _write_jsonl(rich_path, rich_rows)
    manifest = {
        "judge_records_path": str(judge_path),
        "strict_path": str(strict_path),
        "rich_path": str(rich_path),
        "strict_samples": len(strict_rows),
        "rich_samples": len(rich_rows),
        "skipped_non_exportable": skipped_non_exportable,
        "skipped_invalid": skipped_invalid,
    }
    _write_json(manifest_path, manifest)
    return manifest


def validate_openrlhf_export_file(
    *,
    strict_path: str | Path,
    rich_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    strict_rows = _read_jsonl(Path(strict_path))
    if not strict_rows:
        raise ValueError("strict export is empty")
    for index, row in enumerate(strict_rows, start=1):
        if set(row.keys()) != {"chosen", "rejected"}:
            raise ValueError(f"strict row {index} must contain only chosen and rejected")
        if not str(row.get("chosen") or "").strip():
            raise ValueError(f"strict row {index} has empty chosen")
        if not str(row.get("rejected") or "").strip():
            raise ValueError(f"strict row {index} has empty rejected")

    payload: dict[str, Any] = {"strict_samples": len(strict_rows)}
    if rich_path is not None:
        rich_rows = _read_jsonl(Path(rich_path))
        payload["rich_samples"] = len(rich_rows)
    if manifest_path is not None:
        manifest = _load_json_object(Path(manifest_path))
        payload["manifest"] = manifest
        if int(manifest.get("strict_samples") or 0) != len(strict_rows):
            raise ValueError("manifest strict_samples does not match strict export")
    payload["status"] = "ok"
    return payload


def normalize_judge_payload(payload: dict[str, Any], canonical_chosen: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("judge payload must be an object")
    score = _coerce_score(payload.get("score"))
    confidence = _coerce_score(payload.get("confidence"))
    if score is None or confidence is None:
        raise ValueError("judge score and confidence must be numeric")
    failure_modes = payload.get("failure_modes")
    if not isinstance(failure_modes, list) or not all(isinstance(item, str) for item in failure_modes):
        raise ValueError("judge failure_modes must be a string array")
    reason = str(payload.get("reason") or "").strip()
    chosen_response = str(payload.get("chosen_response") or canonical_chosen).strip()
    rejected_response = str(payload.get("rejected_response") or "").strip()
    export_decision = payload.get("export_decision")
    if not isinstance(export_decision, dict):
        raise ValueError("judge export_decision must be an object")
    severe = {item.strip().lower() for item in failure_modes} & SEVERE_FAILURE_MODES
    label = "accept" if score >= 0.75 and confidence >= 0.60 and not severe else "reject"
    exportable = bool(export_decision.get("openrlhf_preference")) and label == "accept" and bool(rejected_response)
    if not chosen_response:
        raise ValueError("judge chosen_response is empty")
    if label == "accept" and not rejected_response:
        raise ValueError("accepted judge output requires rejected_response")
    return {
        "chosen_response": chosen_response,
        "rejected_response": rejected_response,
        "judge": {
            "score": score,
            "confidence": confidence,
            "label": label,
            "failure_modes": failure_modes,
            "reason": reason,
        },
        "export_decision": {"openrlhf_preference": exportable},
    }


def _invalid_judge_payload(error: str, canonical_chosen: str) -> dict[str, Any]:
    return {
        "chosen_response": canonical_chosen,
        "rejected_response": "",
        "judge": {
            "score": 0.0,
            "confidence": 0.0,
            "label": "reject",
            "failure_modes": ["invalid_format"],
            "reason": error or "invalid judge output",
        },
        "export_decision": {"openrlhf_preference": False},
    }


def _judge_record_with_openai(record: dict[str, Any], model: str, reasoning_effort: str) -> dict[str, Any]:
    llm = openai_llm(
        model=model,
        reasoning_effort=reasoning_effort,
        reasoning_summary="auto",
        tool_choice="required",
    )
    instructions = _judge_instructions()
    prompt = json.dumps(
        {
            "task": "Judge one OnlyTeam decision record for RL preference export.",
            "decision_record": record,
            "required_dimensions": [
                "task_progress",
                "decision_quality",
                "thinking_quality",
                "evidence_grounding",
                "tool_fit",
                "phase_compliance",
                "failure_risk",
            ],
            "output_schema": {
                "score": "0.0-1.0 float",
                "confidence": "0.0-1.0 float",
                "reason": "short string",
                "failure_modes": ["string"],
                "chosen_response": "string",
                "rejected_response": "string",
                "export_decision": {"openrlhf_preference": True},
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    stream = llm(input_items=[{"role": "user", "content": prompt}], tools=[], instructions=instructions)
    text = _collect_response_text(stream)
    return _parse_json_object_from_text(text)


def _judge_instructions() -> str:
    return (
        "You are a strict frontier judge for RL preference data construction.\n"
        "Use only the supplied decision record.\n"
        "Do not invent missing context.\n"
        "chosen_response must stay very close to the original trajectory and only lightly normalize it.\n"
        "rejected_response must be a plausible but worse alternative action or plan.\n"
        "If evidence is weak, reduce score/confidence and set openrlhf_preference to false.\n"
        "Return one JSON object only. No markdown. No code fences."
    )


def _collect_response_text(stream: Any) -> str:
    delta_parts: list[str] = []
    item_texts: list[str] = []
    response_output: list[Any] = []
    for event in stream:
        event_type = getattr(event, "type", "")
        if event_type == "response.output_text.delta":
            delta_parts.append(str(getattr(event, "delta", "") or ""))
        elif event_type == "response.output_item.done" and hasattr(event, "item"):
            item_text = _extract_message_text(getattr(event, "item"))
            if item_text:
                item_texts.append(item_text)
        elif event_type in {"response.completed", "response.done"} and hasattr(event, "response"):
            response_output = list(getattr(event.response, "output", []) or [])
    text = "".join(delta_parts).strip()
    if text:
        return text
    if item_texts:
        return "\n".join(part for part in item_texts if part).strip()
    if response_output:
        combined = "\n".join(_extract_message_text(item) for item in response_output)
        if combined.strip():
            return combined.strip()
    raise ValueError("judge returned no assistant text")


def _extract_message_text(item: Any) -> str:
    if getattr(item, "type", None) != "message":
        return ""
    content = getattr(item, "content", None)
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in {"output_text", "text", "input_text"}:
            parts.append(str(block.get("text", "")))
        elif hasattr(block, "text"):
            parts.append(str(getattr(block, "text", "")))
    return "".join(parts).strip()


def _build_training_prompt(record: dict[str, Any]) -> str:
    state = record.get("state") or {}
    lines = []
    if str(state.get("goal_hint") or "").strip():
        lines.append(f"Goal: {state['goal_hint']}")
    if str(state.get("recent_user_message") or "").strip():
        lines.append(f"Current user request: {state['recent_user_message']}")
    if str(state.get("context_snippet") or "").strip():
        lines.append(f"Context: {state['context_snippet']}")
    return "\n".join(lines).strip()


def _build_canonical_chosen(record: dict[str, Any]) -> str:
    thinking = str(((record.get("thinking") or {}).get("clean")) or "").strip()
    decision = record.get("decision") or {}
    observation = record.get("observation") or {}
    action = f"{decision.get('tool_name')}({json.dumps(decision.get('tool_arguments_json') or {}, ensure_ascii=False)})"
    parts = []
    if thinking:
        parts.append(f"<thinking>\n{thinking}\n</thinking>")
    parts.append(f"<action>\n{action}\n</action>")
    if str(observation.get("tool_output") or "").strip():
        parts.append(f"<observation>\n{str(observation['tool_output']).strip()}\n</observation>")
    return "\n".join(parts).strip()


def _recent_user_message(records: list[dict[str, Any]], index: int) -> str:
    for cursor in range(index - 1, -1, -1):
        record = records[cursor]
        if record.get("role") == "user":
            text = str(record.get("content") or "").strip()
            if text:
                return text
    return ""


def _first_nonempty_user_message(records: list[dict[str, Any]]) -> str:
    for record in records:
        if record.get("role") == "user":
            text = str(record.get("content") or "").strip()
            if text:
                return text
    return ""


def _find_output_record(records: list[dict[str, Any]], call_id: str, start_index: int) -> tuple[dict[str, Any] | None, int]:
    for index in range(start_index, len(records)):
        record = records[index]
        if record.get("type") == "function_call_output" and str(record.get("call_id") or "") == call_id:
            return record, index
    return None, -1


def _nearest_response_id(records: list[dict[str, Any]], call_index: int, output_index: int) -> str | None:
    for index in range(call_index, min(output_index + 3, len(records))):
        record = records[index]
        if record.get("type") == "token_usage" and record.get("response_id"):
            return str(record.get("response_id"))
    for index in range(call_index - 1, -1, -1):
        record = records[index]
        if record.get("type") == "token_usage" and record.get("response_id"):
            return str(record.get("response_id"))
    return None


def _build_context_snippet(recent_user_message: str, tool_name: str) -> str:
    snippet = recent_user_message.strip()
    if len(snippet) > 220:
        snippet = snippet[:220].rstrip() + "..."
    if snippet:
        return f"user asks: {snippet}; next tool: {tool_name}"
    return f"next tool: {tool_name}"


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _parse_json_object(value: str) -> dict[str, Any]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _parse_json_object_from_text(text: str) -> dict[str, Any]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge output is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError("judge output must be a JSON object")
    return loaded


def _coerce_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return round(score, 6)


def _source_mem_path(workspace: str | Path | None, source_run_id: str) -> Path:
    return Path(workspace or Path.cwd()).resolve() / ".mem" / source_run_id / "only_team.json"


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(loaded, list):
        raise ValueError(f"expected JSON array in {path}")
    return [item for item in loaded if isinstance(item, dict)]


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON file: {path}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"expected JSON object in {path}")
    return loaded


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{index}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"invalid JSONL object at {path}:{index}")
        rows.append(row)
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "build_decision_records_file",
    "export_openrlhf_preference_file",
    "judge_decision_records_file",
    "list_dataset_runs",
    "normalize_judge_payload",
    "read_mem_run",
    "validate_openrlhf_export_file",
]
