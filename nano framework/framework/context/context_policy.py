"""Static context policy knobs."""

from __future__ import annotations

from dataclasses import dataclass, field


DEFAULT_COMPACTABLE_TOOLS = frozenset(
    {
        "read_file",
        "read_many",
        "grep_search",
        "list_directory",
        "run_powershell",
        "write_file",
        "append_file",
        "apply_diff",
        "submit_structured_result",
        "confirm_research_state",
    }
)

CONTROL_TOOLS = frozenset({"decide", "new_decide", "chat"})


@dataclass(frozen=True)
class ContextPolicy:
    """Context defaults expressed for local record dictionaries."""

    context_window_tokens: int = 160_000
    max_summary_output_tokens: int = 20_000
    autocompact_buffer_tokens: int = 13_000
    manual_compact_buffer_tokens: int = 3_000
    max_consecutive_compact_failures: int = 3
    keep_recent_tool_turns: int = 50
    keep_recent_tool_results: int = 60
    keep_recent_tool_argument_calls: int = 20
    min_preserved_tokens: int = 10_000
    min_preserved_text_messages: int = 5
    max_preserved_tokens: int = 40_000
    argument_preview_chars: int = 2_000
    old_tool_output_preview_chars: int = 2_000
    compactable_tools: frozenset[str] = field(default_factory=lambda: DEFAULT_COMPACTABLE_TOOLS)
    control_tools: frozenset[str] = field(default_factory=lambda: CONTROL_TOOLS)

    @property
    def effective_context_window_tokens(self) -> int:
        return self.context_window_tokens - min(self.max_summary_output_tokens, self.context_window_tokens)

    @property
    def autocompact_threshold_tokens(self) -> int:
        return self.effective_context_window_tokens - self.autocompact_buffer_tokens

    @property
    def blocking_threshold_tokens(self) -> int:
        return self.effective_context_window_tokens - self.manual_compact_buffer_tokens


DEFAULT_CONTEXT_POLICY = ContextPolicy()


__all__ = [
    "CONTROL_TOOLS",
    "ContextPolicy",
    "DEFAULT_COMPACTABLE_TOOLS",
    "DEFAULT_CONTEXT_POLICY",
]

