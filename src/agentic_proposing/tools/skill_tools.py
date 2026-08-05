from __future__ import annotations

from typing import Any

from agentic_proposing.core.models import SideEffect, ToolResult
from agentic_proposing.skills.models import CompositeSpec, SkillManifest
from .base import BaseTool, ToolContext, ToolSpec


class SkillSearchTool(BaseTool):
    spec = ToolSpec(
        name="skill_search",
        description="Search Skill discovery metadata without loading full SKILL.md bodies.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30},
            },
            "required": ["query"],
        },
        deferred=False,
        tags=["skill", "discovery", "progressive-disclosure"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        registry = context.services["skill_registry"]
        matches = registry.search(str(arguments["query"]), int(arguments.get("limit", 10)))
        return ToolResult(call_id, self.spec.name, f"Found {len(matches)} matching skills", {"matches": matches})


class SkillLoadTool(BaseTool):
    spec = ToolSpec(
        name="skill_load",
        description="Load selected SKILL.md procedures into future model context; metadata remains separately indexed.",
        input_schema={
            "type": "object",
            "properties": {"skill_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["skill_ids"],
        },
        deferred=False,
        tags=["skill", "load", "progressive-disclosure"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        registry = context.services["skill_registry"]
        requested = [str(x) for x in arguments.get("skill_ids", [])]
        missing = [item for item in requested if not registry.has(item)]
        loaded = list(context.state.loaded_skill_ids)
        for skill_id in requested:
            if skill_id not in missing and skill_id not in loaded:
                loaded.append(skill_id)
        active = list(context.state.active_skill_ids)
        enabled_tools: list[str] = []
        for skill_id in requested:
            if skill_id not in missing and skill_id not in active:
                active.append(skill_id)
            if skill_id not in missing:
                for tool_name in registry.get_manifest(skill_id).tools:
                    if tool_name not in enabled_tools:
                        enabled_tools.append(tool_name)
        return ToolResult(
            call_id,
            self.spec.name,
            f"Loaded skills: {[item for item in requested if item not in missing]}",
            data={"loaded": loaded, "missing": missing, "enabled_tools": enabled_tools},
            is_error=bool(missing),
            state_updates={
                "loaded_skill_ids": loaded,
                "active_skill_ids": active,
                "enable_tool": enabled_tools,
                "stage": "select",
            },
        )


class SkillComposeTool(BaseTool):
    spec = ToolSpec(
        name="skill_compose",
        description=(
            "Compile parent skill instances into a staged first-class composite Skill. Reusing one skill is allowed "
            "only with distinct instance IDs and bindings. The graph must be acyclic and state a novelty claim."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "spec": {
                    "type": "object",
                    "properties": {
                        "skill_id": {"type": "string"},
                        "name": {"type": "string"},
                        "version": {"type": "string"},
                        "description": {"type": "string"},
                        "intent": {"type": "string"},
                        "instances": {"type": "array", "items": {"type": "object"}},
                        "edges": {"type": "array", "items": {"type": "object"}},
                        "inputs": {"type": "array", "items": {"type": "string"}},
                        "outputs": {"type": "array", "items": {"type": "string"}},
                        "procedure": {"type": "array", "items": {"type": "string"}},
                        "novelty_claim": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "skill_id", "description", "intent", "instances", "edges", "inputs", "outputs",
                        "procedure", "novelty_claim"
                    ],
                }
            },
            "required": ["spec"],
        },
        side_effect=SideEffect.REGISTRY,
        read_only=False,
        deferred=False,
        tags=["skill", "compose", "recursive", "graph"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        registry = context.services["skill_registry"]
        spec = CompositeSpec.from_dict(dict(arguments["spec"]))
        destination = registry.stage_composite(spec)
        graph = {
            "skill_id": spec.skill_id,
            "instances": [
                {"instance_id": x.instance_id, "skill_id": x.skill_id, "bind": x.bind, "role": x.role}
                for x in spec.instances
            ],
            "edges": [{"from": x.source, "to": x.target, "data_type": x.data_type} for x in spec.edges],
        }
        return ToolResult(
            call_id,
            self.spec.name,
            f"Staged composite skill {spec.skill_id}",
            data={"staged_path": str(destination), "composition_graph": graph},
            state_updates={
                "composition_graph": graph,
                "stage": "compose",
                "staged_composite_skill": spec.skill_id,
            },
        )


class SkillStageTool(BaseTool):
    spec = ToolSpec(
        name="skill_stage",
        description="Stage a newly extracted atomic or emergent Skill with full lineage; staging does not publish it.",
        input_schema={
            "type": "object",
            "properties": {
                "manifest": {"type": "object"},
                "body": {"type": "string"},
                "lineage": {"type": "object"},
                "tests": {"type": "object"},
            },
            "required": ["manifest", "body", "lineage"],
        },
        side_effect=SideEffect.REGISTRY,
        read_only=False,
        deferred=True,
        tags=["skill", "extract", "stage", "lineage"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        registry = context.services["skill_registry"]
        manifest = SkillManifest.from_dict(dict(arguments["manifest"]))
        duplicates = registry.duplicate_report(manifest)
        path = registry.stage_atomic(
            manifest,
            str(arguments["body"]),
            dict(arguments["lineage"]),
            dict(arguments.get("tests", {})),
        )
        return ToolResult(
            call_id,
            self.spec.name,
            f"Staged skill {manifest.skill_id}; promotion requires replay and validation",
            data={"staged_path": str(path), "duplicates": duplicates},
            state_updates={"staged_skill": manifest.skill_id, "stage": "mine"},
        )


class SkillLineageTool(BaseTool):
    spec = ToolSpec(
        name="skill_lineage",
        description="Inspect a skill's recursive parents and source provenance without loading all parent bodies.",
        input_schema={
            "type": "object",
            "properties": {
                "skill_id": {"type": "string"},
                "max_depth": {"type": "integer", "minimum": 0, "maximum": 20},
            },
            "required": ["skill_id"],
        },
        deferred=True,
        tags=["skill", "lineage", "provenance", "recursive"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        registry = context.services["skill_registry"]
        lineage = registry.lineage(str(arguments["skill_id"]), int(arguments.get("max_depth", 8)))
        return ToolResult(call_id, self.spec.name, "Skill lineage resolved", {"lineage": lineage})

