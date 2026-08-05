from __future__ import annotations

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
import json
import traceback

from agentic_proposing.core.hooks import HookBus, HookEvent, PermissionPolicy
from agentic_proposing.core.models import Event, RunState, SideEffect, ToolCall, ToolResult, jsonable
from agentic_proposing.core.store import RunStore, WorkspaceLayout


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] = field(default_factory=dict)
    side_effect: SideEffect = SideEffect.NONE
    read_only: bool = True
    parallelizable: bool = False
    deferred: bool = True
    tags: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "side_effect": self.side_effect.value,
            "read_only": self.read_only,
            "tags": self.tags,
        }

    def as_model_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


@dataclass(slots=True)
class ToolContext:
    layout: WorkspaceLayout
    run_store: RunStore
    state: RunState
    services: dict[str, Any] = field(default_factory=dict)

    @property
    def workspace(self) -> Path:
        return self.layout.root


class BaseTool(ABC):
    spec: ToolSpec

    @abstractmethod
    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        if tool.spec.name in self._tools:
            raise ValueError(f"Duplicate tool: {tool.spec.name}")
        self._tools[tool.spec.name] = tool

    def upsert(self, tool: BaseTool) -> None:
        self._tools[tool.spec.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        terms = {part.lower() for part in query.replace("_", " ").split() if part}
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for name, tool in self._tools.items():
            haystack = " ".join([name, tool.spec.description, *tool.spec.tags]).lower()
            score = sum(3 if term in name.lower() else 1 for term in terms if term in haystack)
            if not terms or score:
                ranked.append((score, name, tool.spec.summary()))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        return [item for _, _, item in ranked[:limit]]

    def model_tools(self, enabled: Iterable[str], include_non_deferred: bool = True) -> list[dict[str, Any]]:
        enabled_set = set(enabled)
        specs = []
        for tool in self._tools.values():
            if (include_non_deferred and not tool.spec.deferred) or tool.spec.name in enabled_set:
                specs.append(tool.spec.as_model_tool())
        return specs

    def names(self) -> list[str]:
        return sorted(self._tools)


class ToolDispatcher:
    def __init__(
        self,
        registry: ToolRegistry,
        hooks: HookBus | None = None,
        policy: PermissionPolicy | None = None,
        max_parallel: int = 8,
        max_inline_chars: int = 24_000,
        permission_resolver: Callable[[dict[str, Any]], bool] | None = None,
    ):
        self.registry = registry
        self.hooks = hooks or HookBus()
        self.policy = policy or PermissionPolicy()
        self.max_parallel = max_parallel
        self.max_inline_chars = max_inline_chars
        self.permission_resolver = permission_resolver

    def _bound_result(self, result: ToolResult, context: ToolContext) -> ToolResult:
        serialized = json.dumps(jsonable(result.data), ensure_ascii=False, default=str)
        if len(result.content) + len(serialized) <= self.max_inline_chars:
            return result
        full = {
            "content": result.content,
            "data": result.data,
            "is_error": result.is_error,
            "terminal": result.terminal,
        }
        ref = context.run_store.write_artifact(
            f"{result.call_id}-{result.tool_name}.json",
            full,
            "tool_outputs",
        )
        if ref not in result.artifact_refs:
            result.artifact_refs.append(ref)
        result.content = result.content[:4000] + "\n[Full tool result stored as an artifact.]"
        result.data = {
            "truncated": True,
            "artifact_ref": ref,
            "preview": serialized[: max(1000, self.max_inline_chars - len(result.content))],
        }
        return result

    def _execute_one(self, call: ToolCall, context: ToolContext) -> ToolResult:
        try:
            tool = self.registry.get(call.name)
        except KeyError as exc:
            return ToolResult(call.call_id, call.name, str(exc), is_error=True)
        permission = self.policy.decide(
            tool.spec.name, tool.spec.side_effect.value, tool.spec.read_only
        )
        if (
            not permission.allow
            and permission.metadata.get("requires_approval")
            and self.permission_resolver is not None
        ):
            permission.allow = bool(
                self.permission_resolver(
                    {
                        "tool_name": tool.spec.name,
                        "side_effect": tool.spec.side_effect.value,
                        "read_only": tool.spec.read_only,
                        "arguments": dict(call.arguments),
                    }
                )
            )
            if permission.allow:
                permission.reason = "approved by permission resolver"
        if not permission.allow:
            return ToolResult(call.call_id, call.name, permission.reason, is_error=True)
        payload = {
            "run_id": context.state.run_id,
            "tool_name": call.name,
            "input": dict(call.arguments),
            "call_id": call.call_id,
        }
        hook = self.hooks.emit(HookEvent.PRE_TOOL, payload)
        if not hook.allow:
            return ToolResult(call.call_id, call.name, hook.reason, is_error=True)
        arguments = hook.updated_input if hook.updated_input is not None else call.arguments
        context.run_store.events.append(Event("tool_requested", context.state.run_id, payload))
        try:
            result = tool.invoke(dict(arguments), context, call.call_id)
        except Exception as exc:  # Tool failures must return to the agent loop.
            result = ToolResult(
                call.call_id,
                call.name,
                f"{type(exc).__name__}: {exc}",
                data={"traceback": traceback.format_exc(limit=8)},
                is_error=True,
            )
            self.hooks.emit(HookEvent.TOOL_FAILURE, {**payload, "error": result.content})
        result = self._bound_result(result, context)
        if result.state_updates:
            context.state.apply_updates(result.state_updates)
        for ref in result.artifact_refs:
            if ref not in context.state.artifact_refs:
                context.state.artifact_refs.append(ref)
        self.hooks.emit(HookEvent.POST_TOOL, {**payload, "result": jsonable(result)})
        context.run_store.events.append(
            Event("tool_failed" if result.is_error else "tool_succeeded", context.state.run_id, jsonable(result))
        )
        return result

    def execute(self, calls: list[ToolCall], context: ToolContext) -> list[ToolResult]:
        if len(calls) <= 1:
            return [self._execute_one(call, context) for call in calls]
        results: dict[str, ToolResult] = {}
        concurrent_calls: list[ToolCall] = []
        sequential_calls: list[ToolCall] = []
        for call in calls:
            try:
                spec = self.registry.get(call.name).spec
            except KeyError:
                sequential_calls.append(call)
                continue
            if spec.read_only and spec.parallelizable:
                concurrent_calls.append(call)
            else:
                sequential_calls.append(call)
        if concurrent_calls:
            with ThreadPoolExecutor(max_workers=min(self.max_parallel, len(concurrent_calls))) as pool:
                futures = {pool.submit(self._execute_one, call, context): call for call in concurrent_calls}
                for future in as_completed(futures):
                    result = future.result()
                    results[result.call_id] = result
        for call in sequential_calls:
            result = self._execute_one(call, context)
            results[result.call_id] = result
        return [results[call.call_id] for call in calls]

