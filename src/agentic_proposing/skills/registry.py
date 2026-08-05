from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json
import os
import shutil

from agentic_proposing.core.store import WorkspaceLayout, atomic_write_text
from .dedup import find_duplicates
from .models import CompositeSpec, SkillManifest, SkillPackage


class SkillRegistry:
    """Metadata-first registry. SKILL.md bodies are not read during indexing."""

    def __init__(self, layout: WorkspaceLayout):
        self.layout = layout
        self._packages: dict[str, SkillPackage] = {}
        self.refresh()

    def refresh(self) -> None:
        self._packages.clear()
        for kind in ("atomic", "composite"):
            root = self.layout.skills / kind
            if not root.exists():
                continue
            for manifest_path in root.glob("*/manifest.yaml"):
                manifest = SkillManifest.load(manifest_path)
                if manifest.skill_id in self._packages:
                    raise ValueError(f"Duplicate skill_id in registry: {manifest.skill_id}")
                self._packages[manifest.skill_id] = SkillPackage(manifest_path.parent, manifest)

    def has(self, skill_id: str) -> bool:
        return skill_id in self._packages

    def get_manifest(self, skill_id: str) -> SkillManifest:
        try:
            return self._packages[skill_id].manifest
        except KeyError as exc:
            raise KeyError(f"Skill not found: {skill_id}") from exc

    def get_package(self, skill_id: str) -> SkillPackage:
        try:
            return self._packages[skill_id]
        except KeyError as exc:
            raise KeyError(f"Skill not found: {skill_id}") from exc

    def discovery_summaries(self, skill_ids: list[str] | None = None) -> list[dict[str, Any]]:
        ids = skill_ids or sorted(self._packages)
        return [self._packages[item].manifest.discovery_summary() for item in ids if item in self._packages]

    def load_for_context(self, skill_id: str, max_chars: int) -> str:
        return self.get_package(skill_id).body_excerpt(max_chars)

    def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        terms = {item.lower() for item in query.replace("_", " ").replace("-", " ").split() if item}
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for skill_id, package in self._packages.items():
            manifest = package.manifest
            text = " ".join(
                [
                    skill_id,
                    manifest.description,
                    manifest.intent,
                    *manifest.triggers,
                    *manifest.inputs,
                    *manifest.outputs,
                    *manifest.tags,
                ]
            ).lower()
            score = sum(4 if term in skill_id.lower() else 1 for term in terms if term in text)
            if not terms or score:
                ranked.append((score, skill_id, manifest.discovery_summary()))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [summary for _, _, summary in ranked[:limit]]

    def stage_atomic(
        self,
        manifest: SkillManifest,
        body: str,
        lineage: dict[str, Any],
        tests: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Path:
        destination = self.layout.staging / "skills" / f"{manifest.skill_id}@{manifest.version}"
        if destination.exists():
            raise FileExistsError(f"Staged skill already exists: {destination}")
        destination.mkdir(parents=True)
        manifest.status = "staged"
        atomic_write_text(
            destination / "manifest.yaml",
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        )
        atomic_write_text(destination / "SKILL.md", body)
        atomic_write_text(
            destination / "lineage.json",
            json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True),
        )
        tests_dir = destination / "tests"
        tests_dir.mkdir()
        for name, rows in (tests or {}).items():
            with (tests_dir / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        for directory in ("tools", "references", "verification"):
            (destination / directory).mkdir()
        return destination

    def stage_composite(self, spec: CompositeSpec) -> Path:
        from .compiler import SkillCompiler

        destination = self.layout.staging / "skills" / f"{spec.skill_id}@{spec.version}"
        return SkillCompiler(self).compile_to(spec, destination, status="staged")

    def duplicate_report(self, manifest: SkillManifest) -> list[dict[str, Any]]:
        return [
            {"skill_id": match.skill_id, "score": match.score, "relation": match.relation}
            for match in find_duplicates(manifest, [pkg.manifest for pkg in self._packages.values()])
        ]

    def promote(
        self,
        staged_name: str,
        validation: dict[str, Any],
        *,
        replace_existing: bool = False,
    ) -> Path:
        source = self.layout.staging / "skills" / staged_name
        if not source.exists():
            raise FileNotFoundError(source)
        manifest_path = source / "manifest.yaml"
        manifest = SkillManifest.load(manifest_path)
        if not validation.get("passed", False):
            raise ValueError("Cannot promote skill without passed validation")
        duplicate_report = self.duplicate_report(manifest)
        blocking = [
            item for item in duplicate_report
            if item["relation"] == "duplicate_of"
            and not (replace_existing and item["skill_id"] == manifest.skill_id)
        ]
        if blocking and not validation.get("allow_duplicate", False):
            raise ValueError(f"Duplicate skill blocked: {blocking}")
        destination = self.layout.skills / manifest.kind / manifest.skill_id
        if destination.exists() and not replace_existing:
            raise FileExistsError(f"Registry skill already exists: {manifest.skill_id}")
        archive: Path | None = None
        if destination.exists():
            active = SkillManifest.load(destination / "manifest.yaml")
            if active.version == manifest.version:
                raise ValueError("Replacement version must differ from the active version")
            version_refs = [
                ref for ref in manifest.source_refs
                if ref.get("type") == "versioned_from"
                and ref.get("skill_id") == manifest.skill_id
                and ref.get("version") == active.version
            ]
            if not version_refs:
                raise ValueError("Replacement must declare versioned_from lineage to the active version")
            archive = self.layout.skills / "deprecated" / f"{manifest.skill_id}@{active.version}"
            if archive.exists():
                raise FileExistsError(f"Archived version already exists: {archive}")
        manifest.status = "verified"
        manifest.metrics.update(validation)
        atomic_write_text(
            manifest_path,
            json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        )
        atomic_write_text(
            source / "metrics.json",
            json.dumps(validation, ensure_ascii=False, indent=2, sort_keys=True),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if archive is None:
            os.replace(source, destination)
        else:
            archive.parent.mkdir(parents=True, exist_ok=True)
            os.replace(destination, archive)
            try:
                os.replace(source, destination)
            except Exception:
                os.replace(archive, destination)
                raise
        self.refresh()
        return destination

    def lineage(self, skill_id: str, max_depth: int = 8) -> dict[str, Any]:
        seen: set[str] = set()

        def visit(current: str, depth: int) -> dict[str, Any]:
            if depth > max_depth:
                return {"skill_id": current, "truncated": True}
            if current in seen:
                return {"skill_id": current, "shared_ancestor": True}
            seen.add(current)
            manifest = self.get_manifest(current)
            parents = list(manifest.composition.get("parents", []))
            return {
                "skill_id": current,
                "version": manifest.version,
                "kind": manifest.kind,
                "parents": [visit(parent, depth + 1) for parent in parents if self.has(parent)],
                "source_refs": manifest.source_refs,
            }

        return visit(skill_id, 0)

