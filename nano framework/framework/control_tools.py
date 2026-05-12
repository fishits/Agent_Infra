"""Control-flow tools.

This file only defines runtime control actions for the graph.
"""

from __future__ import annotations

from .tools import tool


def render_decide_rules(edges: dict[str, str]) -> str:
    """Render stable route instructions for decide(...)."""
    lines = [
        "",
        "路由规则：",
        "必须且只能选择一条路由，使用 decide(target=..., content=...)。",
        "只有当目标节点需要从干净的本地会话记忆重新开始时，才使用 new_decide(target=..., content=...)。",
    ]
    for target, rule in edges.items():
        lines.append(f'- decide(target="{target}", content="..."): {rule}')
        lines.append(f'- new_decide(target="{target}", content="..."): {rule}，并清空目标节点的本地记忆后再开始。')
    lines.append('只有确实需要用户输入时，才使用 chat(message="...")。')
    return "\n".join(lines)


@tool(
    description="当前任务完成后跳转到目标节点。",
    params={
        "target": ("string", "目标节点名称，例如 end 或其他节点。"),
        "content": ("string", "传给下一个节点的原因或负载。"),
    },
)
def decide(target: str, content: str) -> str:
    return f"DECIDE:{target}:{content}"


@tool(
    description="跳转到目标节点，并在启动前清空该节点的本地会话记录。",
    params={
        "target": ("string", "目标节点名称，例如 end 或其他节点。"),
        "content": ("string", "传给下一个节点的原因或负载。"),
    },
)
def new_decide(target: str, content: str) -> str:
    return f"NEW_DECIDE:{target}:{content}"


@tool(
    description="向用户请求输入，并暂停当前节点。",
    params={
        "message": ("string", "展示给用户的消息。"),
    },
)
def chat(message: str) -> str:
    return f"CHAT:{message}"


__all__ = ["chat", "decide", "new_decide", "render_decide_rules"]
