from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class HookEvent(str, Enum):
    SESSION_START = "session_start"
    PRE_MODEL = "pre_model"
    POST_MODEL = "post_model"
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"
    TOOL_FAILURE = "tool_failure"
    PRE_COMPACT = "pre_compact"
    PRE_SUBMIT = "pre_submit"
    POST_SUBMIT = "post_submit"
    SKILL_STAGED = "skill_staged"
    SKILL_PROMOTED = "skill_promoted"
    SESSION_END = "session_end"


@dataclass(slots=True)
class HookDecision:
    allow: bool = True
    reason: str = ""
    updated_input: dict[str, Any] | None = None
    additional_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


HookCallback = Callable[[HookEvent, dict[str, Any]], HookDecision | None]


class HookBus:
    def __init__(self):
        self._callbacks: dict[HookEvent, list[HookCallback]] = {event: [] for event in HookEvent}

    def register(self, event: HookEvent, callback: HookCallback) -> None:
        self._callbacks[event].append(callback)

    def emit(self, event: HookEvent, payload: dict[str, Any]) -> HookDecision:
        final = HookDecision()
        for callback in self._callbacks[event]:
            decision = callback(event, payload)
            if decision is None:
                continue
            if decision.updated_input is not None:
                final.updated_input = decision.updated_input
                payload = {**payload, "input": decision.updated_input}
            if decision.additional_context:
                final.additional_context += ("\n" if final.additional_context else "") + decision.additional_context
            final.metadata.update(decision.metadata)
            if not decision.allow:
                final.allow = False
                final.reason = decision.reason or "Blocked by hook"
                break
        return final


@dataclass(slots=True)
class PermissionPolicy:
    allowed: set[str] = field(default_factory=set)
    denied: set[str] = field(default_factory=set)
    ask: set[str] = field(default_factory=set)
    allow_read_only: bool = True
    allow_network: bool = False
    allow_process: bool = False
    allow_registry_writes: bool = True

    def decide(self, tool_name: str, side_effect: str, read_only: bool) -> HookDecision:
        if tool_name in self.denied:
            return HookDecision(False, f"Tool {tool_name} is denied")
        if tool_name in self.ask:
            return HookDecision(
                False,
                f"Tool {tool_name} requires approval",
                metadata={"requires_approval": True},
            )
        if tool_name in self.allowed:
            return HookDecision(True)
        if side_effect == "network" and not self.allow_network:
            return HookDecision(False, "Network tools are disabled by policy")
        if side_effect == "process" and not self.allow_process:
            return HookDecision(False, "Process execution is disabled by policy")
        if side_effect == "registry" and not self.allow_registry_writes:
            return HookDecision(False, "Registry writes are disabled by policy")
        if read_only:
            return HookDecision(
                self.allow_read_only,
                "" if self.allow_read_only else "Read-only tools are disabled by policy",
            )
        return HookDecision(True)

