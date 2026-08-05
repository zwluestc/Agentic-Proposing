from __future__ import annotations

from typing import Any

from agentic_proposing.core.models import SideEffect, ToolResult
from agentic_proposing.skills.management import deprecate_skill, fork_skill_version
from .base import BaseTool, ToolContext, ToolSpec


class SkillPruneTool(BaseTool):
    spec = ToolSpec(
        name="skill_prune",
        description="Remove one or more misaligned Skill instances from the active run set while preserving the registry and audit event.",
        input_schema={
            "type": "object",
            "properties": {
                "skill_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["skill_ids", "reason"],
        },
        side_effect=SideEffect.WORKSPACE,
        read_only=False,
        deferred=False,
        tags=["skill", "prune", "refine", "dynamic-edit"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        remove = {str(item) for item in arguments["skill_ids"]}
        before = list(context.state.active_skill_ids)
        active = [item for item in before if item not in remove]
        loaded = [item for item in context.state.loaded_skill_ids if item not in remove]
        removed = [item for item in before if item in remove]
        return ToolResult(
            call_id,
            self.spec.name,
            f"Pruned active skills: {removed}",
            data={"removed": removed, "reason": str(arguments["reason"])},
            state_updates={
                "active_skill_ids": active,
                "loaded_skill_ids": loaded,
                "stage": "refine",
                "skill_prune_reason": str(arguments["reason"]),
            },
        )


class SkillPromoteTool(BaseTool):
    spec = ToolSpec(
        name="skill_promote",
        description="Promote a staged Skill only after an external replay/validation report marks it passed; duplicate checks still apply.",
        input_schema={
            "type": "object",
            "properties": {
                "staged_name": {"type": "string"},
                "validation": {"type": "object"},
                "replace_existing": {"type": "boolean", "default": False},
            },
            "required": ["staged_name", "validation"],
        },
        side_effect=SideEffect.REGISTRY,
        read_only=False,
        deferred=True,
        tags=["skill", "promote", "validate", "replay"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        path = context.services["skill_registry"].promote(
            str(arguments["staged_name"]),
            dict(arguments["validation"]),
            replace_existing=bool(arguments.get("replace_existing", False)),
        )
        return ToolResult(call_id, self.spec.name, "Skill promoted", data={"path": str(path)})


class SkillDeprecateTool(BaseTool):
    spec = ToolSpec(
        name="skill_deprecate",
        description="Move a verified Skill out of active discovery while preserving its full package, version, reason, and lineage.",
        input_schema={
            "type": "object",
            "properties": {"skill_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["skill_id", "reason"],
        },
        side_effect=SideEffect.REGISTRY,
        read_only=False,
        deferred=True,
        tags=["skill", "deprecate", "version", "manage"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        path = deprecate_skill(
            context.services["skill_registry"], str(arguments["skill_id"]), str(arguments["reason"])
        )
        return ToolResult(call_id, self.spec.name, "Skill deprecated", data={"path": str(path)})


class SkillVersionTool(BaseTool):
    spec = ToolSpec(
        name="skill_version",
        description="Fork an immutable verified Skill into a staged new version with explicit change lineage.",
        input_schema={
            "type": "object",
            "properties": {
                "skill_id": {"type": "string"},
                "new_version": {"type": "string"},
                "change_summary": {"type": "string"},
            },
            "required": ["skill_id", "new_version", "change_summary"],
        },
        side_effect=SideEffect.REGISTRY,
        read_only=False,
        deferred=True,
        tags=["skill", "version", "fork", "manage"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        path = fork_skill_version(
            context.services["skill_registry"],
            str(arguments["skill_id"]),
            str(arguments["new_version"]),
            str(arguments["change_summary"]),
        )
        return ToolResult(call_id, self.spec.name, "Skill version staged", data={"path": str(path)})

