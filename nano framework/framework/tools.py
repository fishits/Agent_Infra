"""FrameWork_light 工具系统。

设计：
- 工具是纯 Python 函数
- 只存储必要的元数据
- 工具执行总是返回字符串
- Agent 负责路由和对话流程

API：
- @tool(...): 声明工具描述和参数元数据
- call_tool(tools, name, args): 执行工具并返回文本
- parse_args(args_str): 解析 JSON 参数字符串为字典
- convert_to_openai_tools(tools): 转换为 OpenAI tools 格式

用法：
- 用 @tool 定义工具
- 用 convert_to_openai_tools() 转换为 OpenAI 格式
- LLM 返回工具调用后，用 call_tool() 执行
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable


Tool = Callable[..., Any]
_HIDDEN_TOOL_PARAMS = {"runtime_context"}
_INTERNAL_TOOL_ARG_KEYS = {
    "历史工具参数",
    "_snipped_for_context",
    "_omitted",
    "[history_tool_args_preview]",
}
_WRITE_LIKE_TOOLS = {"write_file", "append_file", "apply_diff", "apply_patch"}
_HISTORY_PLACEHOLDER_RE = re.compile(r"^\s*历史\s+.+已省略。字符数=\d+\s*$")


def tool(*, description: str, params: dict[str, str] | None = None):
    """Attach lightweight tool metadata to a plain Python function."""

    def decorator(fn: Tool) -> Tool:
        metadata = dict(params or {})
        signature = inspect.signature(fn)
        declared_names = set(metadata.keys())
        function_names = {
            parameter.name
            for parameter in signature.parameters.values()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and parameter.name not in _HIDDEN_TOOL_PARAMS
        }

        unknown_params = sorted(declared_names - function_names)
        missing_params = sorted(function_names - declared_names)
        if unknown_params:
            raise ValueError(f"{fn.__name__}: unknown tool params {unknown_params}")
        if missing_params:
            raise ValueError(f"{fn.__name__}: missing tool params {missing_params}")

        for param_name, meta in metadata.items():
            if not isinstance(meta, tuple) or len(meta) != 2:
                raise ValueError(f"{fn.__name__}.{param_name}: invalid param metadata")

        fn._tool_description = description.strip()  # type: ignore[attr-defined]
        fn._tool_params = metadata  # type: ignore[attr-defined]
        return fn

    return decorator


def call_tool(
    tools: list[Tool],
    name: str,
    args: dict[str, Any] | None = None,
    runtime_context: dict[str, Any] | None = None,
) -> Any:
    """Call a decorated tool by function name."""
    for fn in tools:
        if fn.__name__ == name:
            try:
                payload = _sanitize_tool_payload(dict(args or {}))
            except (ValueError, TypeError) as e:
                return f"参数格式错误: {e}"

            signature = inspect.signature(fn)
            declared_params = getattr(fn, "_tool_params", {})

            if name in _WRITE_LIKE_TOOLS and _contains_history_placeholder(payload):
                return "检测到历史占位内容，已拒绝执行写入。请提供真实参数。"

            unknown_args = sorted(set(payload) - set(declared_params))
            if unknown_args:
                return f"多传参数 {unknown_args[0]}"

            for parameter in signature.parameters.values():
                if parameter.kind not in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    continue
                if parameter.name in _HIDDEN_TOOL_PARAMS:
                    continue
                if parameter.default is inspect.Signature.empty and parameter.name not in payload:
                    return f"缺少参数 {parameter.name}"

            try:
                if "runtime_context" in signature.parameters:
                    payload["runtime_context"] = runtime_context or {}
                return _stringify(fn(**payload))
            except FileNotFoundError:
                return "文件不存在"
            except PermissionError:
                return "权限不足"
            except TimeoutError:
                return "执行超时"
            except ValueError as exc:
                message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
                return message or "参数值错误"
            except TypeError:
                return "工具调用错误"
            except Exception:
                return "工具执行失败"
    return f"工具不存在 {name}"


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _sanitize_tool_payload(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in _INTERNAL_TOOL_ARG_KEYS:
                continue
            cleaned[key] = _sanitize_tool_payload(item)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_tool_payload(item) for item in value]
    return value


def _contains_history_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_HISTORY_PLACEHOLDER_RE.match(value))
    if isinstance(value, dict):
        return any(_contains_history_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_history_placeholder(item) for item in value)
    return False


def parse_args(args_str: str) -> dict[str, Any]:
    """Parse tool arguments from JSON string.
    
    OpenAI API 返回的 arguments 字段是 JSON 字符串。
    这个函数负责解析并返回 dict。
    """
    if not args_str:
        return {}
    
    try:
        return json.loads(args_str)
    except json.JSONDecodeError:
        # JSON 解析失败，返回空 dict
        return {}


#为了适配Openai的格式,之前的版本是适配无工具的messages提示
def convert_to_openai_tools(tools: list[Tool]) -> list[dict]:
    """转换工具为 OpenAI tools 格式。
    
    Args:
        tools: 工具列表
        
    Returns:
        OpenAI tools 格式的列表
    """
    openai_tools = []
    
    type_mapping = {
        "string": "string",
        "integer": "integer",
        "number": "number",
        "boolean": "boolean",
        "array": "array",
        "object": "object",
    }
    
    for fn in tools:
        params = getattr(fn, "_tool_params", {})
        description = getattr(fn, "_tool_description", "")
        
        properties = {}
        required = []
        
        # 检查函数签名，确定必需参数
        signature = inspect.signature(fn)
        for param_name, param_info in signature.parameters.items():
            if param_info.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            if param_name in _HIDDEN_TOOL_PARAMS:
                continue
            
            if param_name in params:
                type_str, desc = params[param_name]
                prop: dict = {
                    "type": type_mapping.get(type_str.lower(), "string"),
                    "description": desc,
                }
                # array 类型必须声明 items，否则 OpenAI 拒绝 schema
                if prop["type"] == "array":
                    prop["items"] = {"type": "string"}
                properties[param_name] = prop
                
                # 没有默认值的参数是必需的
                if param_info.default is inspect.Signature.empty:
                    required.append(param_name)
        
        openai_tools.append({
            "type": "function",
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        })
    
    return openai_tools


__all__ = ["tool", "call_tool", "parse_args", "convert_to_openai_tools"]
