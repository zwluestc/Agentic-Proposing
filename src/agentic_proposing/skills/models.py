from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

from agentic_proposing.core.models import jsonable


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                f"{path} is YAML; install the optional 'yaml' dependency or use JSON-compatible YAML"
            ) from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


@dataclass(slots=True)
class SkillManifest:
    skill_id: str
    name: str
    version: str
    kind: str
    status: str
    description: str
    intent: str
    triggers: list[str] = field(default_factory=list)
    negative_triggers: list[str] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    difficulty_effect: dict[str, Any] = field(default_factory=dict)
    tools: list[str] = field(default_factory=list)
    evidence_contract: dict[str, Any] = field(default_factory=dict)
    composition: dict[str, Any] = field(default_factory=dict)
    source_refs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillManifest":
        required = ("skill_id", "name", "version", "kind", "status", "description", "intent")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"Skill manifest missing: {', '.join(missing)}")
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: data[key] for key in allowed if key in data})

    @classmethod
    def load(cls, path: Path) -> "SkillManifest":
        return cls.from_dict(load_yaml_or_json(path))

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)

    def discovery_summary(self) -> dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "version": self.version,
            "kind": self.kind,
            "status": self.status,
            "description": self.description,
            "intent": self.intent,
            "triggers": self.triggers,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "difficulty_effect": self.difficulty_effect,
            "tags": self.tags,
        }


@dataclass(slots=True)
class SkillInstance:
    instance_id: str
    skill_id: str
    bind: dict[str, str] = field(default_factory=dict)
    role: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillInstance":
        return cls(
            instance_id=str(data["instance_id"]),
            skill_id=str(data["skill_id"]),
            bind={str(k): str(v) for k, v in data.get("bind", {}).items()},
            role=str(data.get("role", "")),
        )


@dataclass(slots=True)
class CompositionEdge:
    source: str
    target: str
    data_type: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositionEdge":
        return cls(
            source=str(data["from"]),
            target=str(data["to"]),
            data_type=str(data.get("data_type", "")),
        )


@dataclass(slots=True)
class CompositeSpec:
    skill_id: str
    name: str
    version: str
    description: str
    intent: str
    instances: list[SkillInstance]
    edges: list[CompositionEdge]
    inputs: list[str]
    outputs: list[str]
    procedure: list[str]
    novelty_claim: list[str] = field(default_factory=list)
    evidence_contract: dict[str, Any] = field(default_factory=dict)
    difficulty_effect: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositeSpec":
        return cls(
            skill_id=str(data["skill_id"]),
            name=str(data.get("name", data["skill_id"])),
            version=str(data.get("version", "0.1.0")),
            description=str(data["description"]),
            intent=str(data["intent"]),
            instances=[SkillInstance.from_dict(x) for x in data.get("instances", [])],
            edges=[CompositionEdge.from_dict(x) for x in data.get("edges", [])],
            inputs=[str(x) for x in data.get("inputs", [])],
            outputs=[str(x) for x in data.get("outputs", [])],
            procedure=[str(x) for x in data.get("procedure", [])],
            novelty_claim=[str(x) for x in data.get("novelty_claim", [])],
            evidence_contract=dict(data.get("evidence_contract", {})),
            difficulty_effect=dict(data.get("difficulty_effect", {})),
            tags=[str(x) for x in data.get("tags", [])],
        )


@dataclass(slots=True)
class SkillPackage:
    root: Path
    manifest: SkillManifest
    _body: str | None = None

    @property
    def body_path(self) -> Path:
        return self.root / "SKILL.md"

    @property
    def body(self) -> str:
        if self._body is None:
            self._body = self.body_path.read_text(encoding="utf-8")
        return self._body

    def body_excerpt(self, max_chars: int) -> str:
        body = self.body
        if len(body) <= max_chars:
            return body
        return body[:max_chars] + "\n\n[Skill body truncated; inspect referenced files on demand.]"

