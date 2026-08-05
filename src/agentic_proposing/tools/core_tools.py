from __future__ import annotations

from typing import Any
import json

from agentic_proposing.core.models import SideEffect, Stage, ToolResult
from .base import BaseTool, ToolContext, ToolSpec


class ToolSearchTool(BaseTool):
    spec = ToolSpec(
        name="tool_search",
        description="Search deferred tools by capability and load selected tool schemas into the next turn.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "load": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30},
            },
            "required": ["query"],
        },
        deferred=False,
        tags=["discovery", "context-management"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        registry = context.services["tool_registry"]
        matches = registry.search(str(arguments.get("query", "")), int(arguments.get("limit", 10)))
        requested = [str(x) for x in arguments.get("load", [])]
        available = {item["name"] for item in matches}
        missing = [name for name in requested if name not in registry.names()]
        loaded = [name for name in requested if name in registry.names()]
        return ToolResult(
            call_id,
            self.spec.name,
            f"Found {len(matches)} tools; loaded {loaded}",
            data={"matches": matches, "loaded": loaded, "missing": missing},
            is_error=bool(missing),
            state_updates={"enable_tool": loaded} if loaded else {},
        )


class WorkingStateEditTool(BaseTool):
    spec = ToolSpec(
        name="working_state_edit",
        description="Persist a concise stage, draft, composition graph, evidence pointers, or failed check in run state.",
        input_schema={
            "type": "object",
            "properties": {
                "stage": {"type": "string", "enum": [item.value for item in Stage]},
                "working_artifact": {"type": "object"},
                "composition_graph": {"type": "object"},
                "active_skill_ids": {"type": "array", "items": {"type": "string"}},
                "failed_check": {"type": "object"},
                "evidence": {"type": ["object", "array"]},
                "decision": {"type": "string"},
            },
        },
        side_effect=SideEffect.WORKSPACE,
        read_only=False,
        deferred=False,
        tags=["state", "checkpoint", "draft", "refine"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        updates: dict[str, Any] = {}
        for key in ("stage", "working_artifact", "composition_graph", "active_skill_ids"):
            if key in arguments:
                updates[key] = arguments[key]
        if "failed_check" in arguments:
            updates["append_failed_check"] = arguments["failed_check"]
        if "evidence" in arguments:
            updates["append_evidence"] = arguments["evidence"]
        if "decision" in arguments:
            context.run_store.events.append(
                __import__("agentic_proposing.core.models", fromlist=["Event"]).Event(
                    "structured_decision", context.state.run_id, {"decision": arguments["decision"]}
                )
            )
        return ToolResult(call_id, self.spec.name, "Working state updated", data=updates, state_updates=updates)

