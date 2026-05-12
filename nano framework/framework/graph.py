"""Graph system.

This file only defines nodes and validates node links.
"""

from __future__ import annotations

from typing import Any, Callable

from .session import create_run_id
from .tools import Tool


def node(
    name: str,
    llm: Callable,
    tools: list[Tool],
    prompt: str | Callable[[dict], str],
    edges: list[str] | dict[str, str] | None = None,
    display_name: str | None = None,
    workdir: str | None = None,
) -> dict:
    """Define one node."""
    edge_list = list(edges.keys()) if isinstance(edges, dict) else list(edges or [])

    return {
        "name": name,
        "display_name": display_name or name,
        "llm": llm,
        "tools": tools,
        "prompt": prompt,
        "edges": edges,
        "_edges": edge_list,
        "workdir": workdir,
    }


def validate_graph(nodes: list[dict]) -> bool:
    """Validate that node names are unique and every edge points to an existing node."""
    errors: list[str] = []
    node_names = [node_data["name"] for node_data in nodes]
    known_names = set(node_names)

    if len(node_names) != len(known_names):
        seen: set[str] = set()
        for node_name in node_names:
            if node_name in seen:
                errors.append(f"Duplicate node name '{node_name}'")
            seen.add(node_name)

    for node_data in nodes:
        node_name = node_data["name"]
        for target in node_data.get("_edges", []):
            if target != "end" and target not in known_names:
                errors.append(f"Node '{node_name}' points to missing target '{target}'")

    if errors:
        print("[ERROR] Graph validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False

    return True


def graph(
    nodes: list[dict],
    run_id: str | None = None,
    start_node: str | None = None,
    state: dict[str, Any] | None = None,
) -> dict:
    """Build one runnable graph object from validated nodes."""
    if not nodes:
        raise ValueError("Graph must contain at least one node.")
    if not validate_graph(nodes):
        raise ValueError("Graph validation failed.")

    base_state = {
        "run_id": run_id or create_run_id(),
        "current_node": start_node or nodes[0]["name"],
    }
    user_state = dict(state or {})
    user_state.pop("run_id", None)
    user_state.pop("current_node", None)
    base_state.update(user_state)

    return {
        "nodes": nodes,
        "state": base_state,
    }


__all__ = ["graph", "node", "validate_graph"]
