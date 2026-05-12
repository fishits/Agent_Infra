"""DatasetTeam single-history phase runner."""

from __future__ import annotations

import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator

from framework.context import build_openai_input_items
from framework.mem import append_items
from framework.tools import call_tool, convert_to_openai_tools, parse_args

from DatasetTeam.acceptance_state import decide_gate, final_gate, pre_tool_gate, record_tool_fact
from DatasetTeam.phases import (
    AGENT_NODE,
    END_PHASE,
    START_PHASE,
    allowed_tool_names,
    known_phase_names,
    next_phases,
    phase_display_name,
)
from DatasetTeam.prompt import build_dynamic_context_items, build_node_prompt


MEMORY_NODE = AGENT_NODE
CONTROL_TOOL_NAMES = {"chat", "decide"}
MAX_NO_TOOL_RETRIES = 3
NO_TOOL_CALL_WARNING = (
    "You must call one available tool on every turn. Continue with an inspection/export tool, "
    "use chat(message=...) if user input is required, or use decide(target=..., reason=...) when the phase gate is met."
)
_thinking_open = False


class DatasetTeamRunner:
    """Run one dataset-construction agent with one shared memory and multiple phases."""

    def __init__(
        self,
        *,
        dataset_team_graph: dict[str, Any] | None = None,
        run_id: str | None = None,
        llm: Callable | None = None,
        workspace: Path | None = None,
        start_phase: str = START_PHASE,
        user_state: dict[str, Any] | None = None,
    ) -> None:
        if dataset_team_graph is None:
            if run_id is None or llm is None:
                raise ValueError("run_id and llm are required when dataset_team_graph is not provided")
            dataset_team_graph = {
                "nodes": [
                    {
                        "name": AGENT_NODE,
                        "display_name": "DatasetTeam",
                        "llm": llm,
                        "tools": [],
                        "prompt": "",
                        "edges": {},
                    }
                ],
                "state": {
                    "run_id": run_id,
                    "current_node": AGENT_NODE,
                    "phase": start_phase,
                    **(user_state or {}),
                },
            }
        self.graph = dataset_team_graph
        self.nodes = {node["name"]: node for node in self.graph["nodes"]}
        self.workspace = (workspace or Path.cwd()).resolve()
        self.state = self.graph["state"]
        self.state.setdefault("workspace", str(self.workspace))
        self.state["current_node"] = AGENT_NODE
        self.state.setdefault("phase", start_phase)
        self.run_id = str(self.state["run_id"])
        if AGENT_NODE not in self.nodes:
            raise ValueError(f"DatasetTeam graph must contain the single agent node {AGENT_NODE!r}")
        if self.phase not in known_phase_names():
            raise ValueError(f"Unknown DatasetTeam phase: {self.phase!r}")

    @property
    def phase(self) -> str:
        return str(self.state.get("phase") or START_PHASE)

    def run(
        self,
        user_message: str = "",
        *,
        event_output: Callable[[dict], None] | None = None,
    ) -> dict[str, Any]:
        interrupted = False

        def signal_handler(signum, frame):
            del signum, frame
            nonlocal interrupted
            interrupted = True
            print("\n\n[WARN] Interrupt received, stopping cleanly...")

        signal.signal(signal.SIGINT, signal_handler)
        try:
            output = event_output or default_event_output
            for event in self.stream(user_message):
                if interrupted:
                    print("[STOP] Execution interrupted")
                    break
                output(event)
        finally:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        return self.state

    def stream(self, user_message: str = "") -> Iterator[dict]:
        pending_message = user_message
        if user_message and not self.state.get("root_user_message"):
            self.state["root_user_message"] = user_message

        while self.phase != END_PHASE:
            node = self.nodes[AGENT_NODE]
            tools = list(node.get("tools") or [])
            yield {
                "type": "phase_start",
                "phase": self.phase,
                "display_name": phase_display_name(self.phase),
                "message": pending_message,
                "root_message": self.state.get("root_user_message", ""),
            }

            context = build_openai_input_items(self.run_id, MEMORY_NODE, include_todo_context=False)
            new_items = []
            if pending_message:
                new_items = [{"role": "user", "content": pending_message}]
                append_items(self.run_id, MEMORY_NODE, new_items)
                yield {"type": "user_message", "content": pending_message}
            dynamic_items = build_dynamic_context_items(node, self.state)
            instructions = build_node_prompt(node, self.state)
            next_phase: str | None = None
            route_reason = ""

            for event in self._turn(
                input_items=context + dynamic_items + new_items,
                llm=node["llm"],
                tools=tools,
                instructions=instructions,
            ):
                event_type = event["type"]
                if event_type in {"response_item", "tool_output_item"}:
                    append_items(self.run_id, MEMORY_NODE, [event["item"]])
                    yield event
                    continue
                if event_type == "response_usage":
                    append_items(self.run_id, MEMORY_NODE, [event["usage"]])
                    yield event
                    continue
                if event_type == "chat_request":
                    yield event
                    pending_message = default_user_input(event["message"])
                    break
                if event_type == "phase_decide":
                    next_phase = event["target"]
                    route_reason = event["reason"]
                    yield event
                    break
                if event_type == "error":
                    yield event
                    return
                yield event
            else:
                pending_message = ""
                continue

            if next_phase:
                previous_phase = self.phase
                self.state["phase"] = next_phase
                self.state["current_node"] = AGENT_NODE
                pending_message = ""
                if next_phase != END_PHASE:
                    yield {
                        "type": "phase_transition",
                        "from": previous_phase,
                        "to": next_phase,
                        "from_display": phase_display_name(previous_phase),
                        "to_display": phase_display_name(next_phase),
                        "reason": route_reason,
                    }
                continue
            pending_message = pending_message or ""
        yield {"type": "completed"}

    def _turn(
        self,
        *,
        input_items: list[dict],
        llm: Callable,
        tools: list[Callable],
        instructions: str,
    ) -> Iterator[dict]:
        current_items = list(input_items)
        for attempt in range(1, MAX_NO_TOOL_RETRIES + 1):
            reasoning_text = ""
            response_stream = llm(
                input_items=current_items,
                tools=convert_to_openai_tools(tools),
                instructions=instructions,
            )
            response_output: list[Any] = []
            tool_calls: dict[str, dict[str, Any]] = {}
            response_id: str | None = None

            for event in response_stream:
                event_type = event.type
                if event_type == "response.reasoning_summary_text.delta":
                    reasoning_text += str(event.delta or "")
                    yield {"type": "reasoning", "content": event.delta}
                elif event_type == "response.output_item.added" and event.item.type == "function_call":
                    tool_calls[event.item.id] = {
                        "name": event.item.name,
                        "call_id": event.item.call_id,
                        "item_id": event.item.id,
                        "arguments": "",
                    }
                    yield {"type": "tool_call_start", "name": event.item.name}
                elif event_type == "response.function_call_arguments.delta":
                    tool_call = tool_calls.get(event.item_id)
                    if tool_call is None:
                        continue
                    tool_call["arguments"] += event.delta
                    yield {"type": "tool_call_args", "name": tool_call["name"], "delta": event.delta}
                elif event_type == "response.output_item.done" and hasattr(event, "item"):
                    if getattr(event.item, "type", None) == "function_call" and reasoning_text:
                        setattr(event.item, "thinking", reasoning_text)
                    response_output.append(event.item)
                    yield {"type": "response_item", "item": event.item}
                elif event_type in {"response.completed", "response.done"} and hasattr(event, "response"):
                    response_id = getattr(event.response, "id", None)
                    usage = getattr(event.response, "usage", None)
                    usage_record = _build_usage_record(usage, response_id)
                    if usage_record is not None:
                        yield {"type": "response_usage", "usage": usage_record}
                    if not response_output and event.response.output:
                        response_output = list(event.response.output)
                        for item in response_output:
                            yield {"type": "response_item", "item": item}

            if not tool_calls:
                yield {"type": "warning", "attempt": attempt, "message": NO_TOOL_CALL_WARNING}
                if attempt < MAX_NO_TOOL_RETRIES:
                    current_items = current_items + [{"role": "user", "content": NO_TOOL_CALL_WARNING}]
                    continue
                yield {
                    "type": "error",
                    "message": "Model failed to call a tool for consecutive attempts. Stopping this phase turn.",
                }
                return

            parsed_tool_calls = [{**tool_call, "args": parse_args(str(tool_call["arguments"]))} for tool_call in tool_calls.values()]
            for tool_call in parsed_tool_calls:
                yield {"type": "tool_call", "name": tool_call["name"], "args": tool_call["args"]}

            executed_results = _execute_tool_calls(
                tools=tools,
                tool_calls=parsed_tool_calls,
                runtime_context={
                    **self.state,
                    "run_id": self.run_id,
                    "node_name": MEMORY_NODE,
                    "phase": self.phase,
                    "current_node": AGENT_NODE,
                    "workspace": str(self.workspace),
                    "allowed_edges": next_phases(self.phase),
                    "allowed_tools": sorted(allowed_tool_names(self.phase)),
                },
            )

            tool_outputs: list[dict[str, str]] = []
            for tool_call, result in zip(parsed_tool_calls, executed_results):
                yield {"type": "tool_result", "name": tool_call["name"], "result": result}
                output_item = {
                    "type": "function_call_output",
                    "call_id": tool_call["call_id"],
                    "output": result,
                }
                tool_outputs.append(output_item)
                yield {"type": "tool_output_item", "item": output_item}
                control = _parse_control_result(result)
                if control is None:
                    continue
                if control["type"] == "chat":
                    yield {
                        "type": "chat_request",
                        "message": control["message"],
                        "context_items": response_output + tool_outputs,
                        "response_id": response_id,
                    }
                    return
                yield {
                    "type": "phase_decide",
                    "target": control["target"],
                    "reason": control["reason"],
                    "context_items": response_output + tool_outputs,
                    "response_id": response_id,
                }
                return
            return


def build_dataset_team_runner(
    *,
    dataset_team_graph: dict[str, Any] | None = None,
    run_id: str | None = None,
    llm: Callable | None = None,
    workspace: Path | None = None,
    start_phase: str = START_PHASE,
    user_state: dict[str, Any] | None = None,
) -> DatasetTeamRunner:
    return DatasetTeamRunner(
        dataset_team_graph=dataset_team_graph,
        run_id=run_id,
        llm=llm,
        workspace=workspace,
        start_phase=start_phase,
        user_state=user_state,
    )


def _execute_tool_calls(
    *,
    tools: list[Callable],
    tool_calls: list[dict[str, Any]],
    runtime_context: dict[str, Any],
) -> list[str]:
    if len(tool_calls) <= 1 or any(call["name"] in CONTROL_TOOL_NAMES for call in tool_calls):
        return [_execute_one_tool_call(tools, tool_call, runtime_context) for tool_call in tool_calls]
    results: list[str | None] = [None] * len(tool_calls)
    with ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
        futures = {
            executor.submit(_execute_one_tool_call, tools, tool_call, runtime_context): index
            for index, tool_call in enumerate(tool_calls)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return [result if result is not None else "tool execution failed" for result in results]


def _execute_one_tool_call(
    tools: list[Callable],
    tool_call: dict[str, Any],
    runtime_context: dict[str, Any],
) -> str:
    phase = str(runtime_context.get("phase") or START_PHASE)
    allowed = allowed_tool_names(phase)
    tool_name = str(tool_call["name"])
    if tool_name not in allowed:
        allowed_text = ", ".join(sorted(allowed))
        return f"TOOL_BLOCKED: current phase={phase} does not allow {tool_name}. Allowed tools: {allowed_text}"

    args = tool_call.get("args") or {}
    if tool_name == "decide":
        gate = decide_gate(
            target=str(args.get("target") or ""),
            phase=phase,
            run_id=runtime_context.get("run_id"),
            node_name=runtime_context.get("node_name"),
            workspace=runtime_context.get("workspace"),
            output_dir=runtime_context.get("output_dir"),
            source_run_id=runtime_context.get("source_run_id"),
        )
        if gate is not None:
            return gate
    elif tool_name == "chat" and phase == "final_report":
        ok, message = final_gate(
            run_id=runtime_context.get("run_id"),
            node_name=runtime_context.get("node_name"),
            workspace=runtime_context.get("workspace"),
            output_dir=runtime_context.get("output_dir"),
        )
        if not ok:
            return message
    else:
        gate = pre_tool_gate(
            tool_name=tool_name,
            phase=phase,
            run_id=runtime_context.get("run_id"),
            node_name=runtime_context.get("node_name"),
        )
        if gate is not None:
            return gate

    result = str(call_tool(tools, tool_name, args, runtime_context=runtime_context))
    record_tool_fact(
        run_id=runtime_context.get("run_id"),
        node_name=runtime_context.get("node_name"),
        workspace=runtime_context.get("workspace"),
        tool_name=tool_name,
        args=args,
        result=result,
    )
    return result


def _parse_control_result(result: str) -> dict[str, str] | None:
    if result.startswith("DECIDE:"):
        target, _, reason = result[7:].partition(":")
        return {"type": "decide", "target": target, "reason": reason}
    if result.startswith("CHAT:"):
        return {"type": "chat", "message": result[5:]}
    return None


def _build_usage_record(usage: object, response_id: str | None) -> dict | None:
    if usage is None:
        return None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    if not all(isinstance(value, int) for value in (input_tokens, output_tokens, total_tokens)):
        return None
    return {
        "type": "token_usage",
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "response_id": response_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def default_user_input(message: str) -> str:
    _print_block("Chat", message)
    return input("[User] ").strip() or "(no input)"


def default_event_output(event: dict) -> None:
    global _thinking_open
    event_type = event.get("type")
    if event_type == "phase_start":
        if _thinking_open:
            print()
            _thinking_open = False
        body = str(event.get("display_name") or event.get("phase") or "Unknown")
        root = str(event.get("root_message") or "").strip()
        message = str(event.get("message") or "").strip()
        if root:
            body += f"\n[Request] {_preview(root, 500)}"
        body += f"\n[Input] {_preview(message, 500) if message else '(continue from context)'}"
        _print_block("Phase", body)
    elif event_type == "reasoning":
        content = str(event.get("content") or "")
        if content:
            if not _thinking_open:
                print("\n[Thinking] ", end="", flush=True)
                _thinking_open = True
            print(content, end="", flush=True)
    elif event_type == "tool_call_start":
        if _thinking_open:
            print()
            _thinking_open = False
        name = str(event.get("name") or "")
        if name not in CONTROL_TOOL_NAMES:
            print(f"\nCalling {name} ...", flush=True)
    elif event_type == "tool_call":
        if _thinking_open:
            print()
            _thinking_open = False
        name = str(event.get("name") or "")
        if name in CONTROL_TOOL_NAMES:
            return
        _print_block("Tool", f"{name}: {_preview(str(event.get('args') or {}), 500)}")
    elif event_type == "tool_result":
        if _thinking_open:
            print()
            _thinking_open = False
        result = str(event.get("result") or "")
        if result.startswith(("CHAT:", "DECIDE:")):
            return
        _print_block("Tool Result", _preview(result, 500))
    elif event_type == "phase_decide":
        if _thinking_open:
            print()
            _thinking_open = False
        _print_block("Decision", f"target={event.get('target')}\nreason={_preview(str(event.get('reason') or ''), 400)}")
    elif event_type == "phase_transition":
        if _thinking_open:
            print()
            _thinking_open = False
        reason = str(event.get("reason") or "").strip()
        text = f"{event.get('from_display')} -> {event.get('to_display')}"
        if reason:
            text += f"\n[Reason] {_preview(reason, 400)}"
        _print_block("Route", text)
    elif event_type == "warning":
        if _thinking_open:
            print()
            _thinking_open = False
        _print_block("Warning", str(event.get("message") or ""))
    elif event_type == "error":
        if _thinking_open:
            print()
            _thinking_open = False
        _print_block("Error", str(event.get("message") or ""))
    elif event_type == "completed":
        if _thinking_open:
            print()
            _thinking_open = False
        _print_block("Done", "Completed")


def _print_block(title: str, body: str) -> None:
    print(f"\n[{title}]\n{body}", flush=True)


def _preview(text: str, limit: int) -> str:
    cleaned = text.replace("\r", "\\r").replace("\n", "\\n")
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + f"... ({len(cleaned) - limit} more chars)"


__all__ = ["DatasetTeamRunner", "MEMORY_NODE", "build_dataset_team_runner", "default_event_output", "default_user_input"]
