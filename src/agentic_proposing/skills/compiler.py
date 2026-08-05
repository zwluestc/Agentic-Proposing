from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json

from agentic_proposing.core.store import atomic_write_text
from .models import CompositeSpec, SkillManifest


class CompositionError(ValueError):
    pass


class SkillCompiler:
    def __init__(self, registry: "SkillRegistry"):
        self.registry = registry

    def validate(self, spec: CompositeSpec) -> list[str]:
        errors: list[str] = []
        instance_ids = [node.instance_id for node in spec.instances]
        if len(instance_ids) != len(set(instance_ids)):
            errors.append("instance_id values must be unique; repeated skill types need distinct instances")
        known_instances = set(instance_ids)
        for node in spec.instances:
            if not self.registry.has(node.skill_id):
                errors.append(f"unknown parent skill: {node.skill_id}")
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in known_instances}
        indegree = {node_id: 0 for node_id in known_instances}
        for edge in spec.edges:
            source_node = edge.source.split(".", 1)[0]
            target_node = edge.target.split(".", 1)[0]
            if source_node not in known_instances:
                errors.append(f"edge source instance not found: {source_node}")
                continue
            if target_node not in known_instances:
                errors.append(f"edge target instance not found: {target_node}")
                continue
            if target_node not in adjacency[source_node]:
                adjacency[source_node].add(target_node)
                indegree[target_node] += 1
        queue = [node for node, degree in indegree.items() if degree == 0]
        visited = 0
        while queue:
            node = queue.pop()
            visited += 1
            for target in adjacency[node]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if visited != len(known_instances):
            errors.append("composition graph contains a cycle")
        if not spec.novelty_claim:
            errors.append("composite skill must state a novelty claim")
        if not spec.procedure:
            errors.append("composite skill requires an executable procedure")
        return errors

    def compile_to(self, spec: CompositeSpec, destination: Path, status: str = "staged") -> Path:
        errors = self.validate(spec)
        if errors:
            raise CompositionError("; ".join(errors))
        destination.mkdir(parents=True, exist_ok=False)
        parents = [node.skill_id for node in spec.instances]
        inherited_tools: list[str] = []
        for parent_id in parents:
            for tool_name in self.registry.get_manifest(parent_id).tools:
                if tool_name not in inherited_tools:
                    inherited_tools.append(tool_name)
        manifest = SkillManifest(
            skill_id=spec.skill_id,
            name=spec.name,
            version=spec.version,
            kind="composite",
            status=status,
            description=spec.description,
            intent=spec.intent,
            inputs=spec.inputs,
            outputs=spec.outputs,
            difficulty_effect=spec.difficulty_effect,
            tools=inherited_tools,
            evidence_contract=spec.evidence_contract,
            composition={
                "parents": parents,
                "instance_count": len(spec.instances),
                "novelty_claim": spec.novelty_claim,
            },
            source_refs=[{"type": "composed_from", "skill_id": parent} for parent in parents],
            tags=spec.tags,
        )
        atomic_write_text(
            destination / "manifest.yaml",
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        )
        graph = {
            "instances": [asdict(item) for item in spec.instances],
            "edges": [
                {"from": edge.source, "to": edge.target, "data_type": edge.data_type}
                for edge in spec.edges
            ],
            "novelty_claim": spec.novelty_claim,
        }
        atomic_write_text(
            destination / "composition.yaml",
            json.dumps(graph, ensure_ascii=False, indent=2, sort_keys=True),
        )
        procedure = "\n".join(f"{index}. {step}" for index, step in enumerate(spec.procedure, 1))
        body = f"""---
name: {spec.skill_id}
kind: composite
version: {spec.version}
status: {status}
---

# Skill of {spec.name}

## Intent

{spec.intent}

## Preconditions and typed inputs

{chr(10).join(f'- `{item}`' for item in spec.inputs) or '- None'}

## Composition procedure

{procedure}

## Outputs

{chr(10).join(f'- `{item}`' for item in spec.outputs) or '- None'}

## Novelty claim

{chr(10).join(f'- {item}' for item in spec.novelty_claim)}

## Failure handling

- If an input binding cannot be grounded, stop and retrieve evidence rather than inventing it.
- If two parent constraints conflict, remove or rebind the conflicting instance and replay validation.
- Repeated occurrences of one parent skill must have distinct instance IDs and bindings.
- Do not inline source facts into this skill; resolve them through the evidence contract at instantiation time.
"""
        atomic_write_text(destination / "SKILL.md", body)
        atomic_write_text(
            destination / "lineage.json",
            json.dumps(
                {
                    "skill_id": spec.skill_id,
                    "version": spec.version,
                    "parents": parents,
                    "graph": graph,
                    "derivation": "recursive_composition",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        )
        for directory in ("tools", "tests", "references", "verification"):
            (destination / directory).mkdir(exist_ok=True)
        return destination


from .registry import SkillRegistry  # noqa: E402  (runtime type without circular import at definition)

