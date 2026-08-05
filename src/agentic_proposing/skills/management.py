from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os
import shutil
import time

from agentic_proposing.core.store import atomic_write_text
from .models import SkillManifest
from .registry import SkillRegistry


def deprecate_skill(registry: SkillRegistry, skill_id: str, reason: str) -> Path:
    package = registry.get_package(skill_id)
    manifest = package.manifest
    manifest.status = "deprecated"
    manifest.metrics["deprecation_reason"] = reason
    manifest.metrics["deprecated_at"] = time.time()
    atomic_write_text(
        package.root / "manifest.yaml",
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
    )
    destination = registry.layout.skills / "deprecated" / f"{skill_id}@{manifest.version}"
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(package.root, destination)
    registry.refresh()
    return destination


def fork_skill_version(
    registry: SkillRegistry,
    skill_id: str,
    new_version: str,
    change_summary: str,
) -> Path:
    package = registry.get_package(skill_id)
    destination = registry.layout.staging / "skills" / f"{skill_id}@{new_version}"
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(package.root, destination)
    manifest = SkillManifest.load(destination / "manifest.yaml")
    old_version = manifest.version
    manifest.version = new_version
    manifest.status = "staged"
    manifest.source_refs.append(
        {
            "type": "versioned_from",
            "skill_id": skill_id,
            "version": old_version,
            "change_summary": change_summary,
        }
    )
    atomic_write_text(
        destination / "manifest.yaml",
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
    )
    lineage_path = destination / "lineage.json"
    lineage = json.loads(lineage_path.read_text(encoding="utf-8")) if lineage_path.exists() else {}
    lineage["versioned_from"] = {
        "skill_id": skill_id,
        "version": old_version,
        "change_summary": change_summary,
    }
    atomic_write_text(lineage_path, json.dumps(lineage, ensure_ascii=False, indent=2, sort_keys=True))
    return destination

