"""Declarative graph API for OnlyTeam."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

from framework.session import create_run_id
from framework.tools import Tool

from OnlyTeam.phase_runtime.runner import MEMORY_NODE, OnlyTeamRunner, default_event_output


def node(
    *,
    name: str,
    llm: Callable,
    tools: list[Tool],
    prompt: str | Callable[[dict[str, Any]], str],
    edges: dict[str, str] | None = None,
    display_name: str | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "display_name": display_name or name,
        "llm": llm,
        "tools": tools,
        "prompt": prompt,
        "edges": dict(edges or {}),
    }


def graph(
    *,
    nodes: list[dict[str, Any]],
    start_node: str,
    run_id: str | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not nodes:
        raise ValueError("OnlyTeam graph requires at least one node")
    names = [item["name"] for item in nodes]
    if len(names) != len(set(names)):
        raise ValueError("OnlyTeam graph node names must be unique")
    known = set(names)
    if start_node not in known:
        raise ValueError(f"start_node {start_node!r} is not in graph nodes")
    for item in nodes:
        for target in item.get("edges", {}):
            if target != "end" and target not in known:
                raise ValueError(f"node {item['name']!r} routes to missing node {target!r}")

    user_state = dict(state or {})
    user_state.pop("run_id", None)
    user_state.pop("current_node", None)
    return {
        "nodes": nodes,
        "state": {
            "run_id": run_id or create_run_id("onlyteam"),
            "current_node": start_node,
            **user_state,
        },
    }


def run(
    only_team_graph: dict[str, Any],
    *,
    user_message: str = "",
    stream: bool = False,
    event_output: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    del stream
    runner = OnlyTeamRunner(
        only_team_graph=only_team_graph,
        workspace=Path(str(only_team_graph["state"].get("workspace") or Path.cwd())),
    )
    return runner.run(user_message=user_message, event_output=event_output or default_event_output)


def stream(
    only_team_graph: dict[str, Any],
    *,
    user_message: str = "",
) -> Iterator[dict]:
    runner = OnlyTeamRunner(
        only_team_graph=only_team_graph,
        workspace=Path(str(only_team_graph["state"].get("workspace") or Path.cwd())),
    )
    yield from runner.stream(user_message=user_message)


__all__ = ["MEMORY_NODE", "graph", "node", "run", "stream"]
