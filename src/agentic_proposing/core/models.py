from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import json
import time
import uuid


class Stage(str, Enum):
    DISCOVER = "discover"
    SELECT = "select"
    COMPOSE = "compose"
    DRAFT = "draft"
    CHECK = "check"
    REFINE = "refine"
    FINALIZE = "finalize"
    MINE = "mine"
    PROMOTE = "promote"
    COMPLETE = "complete"
    FAILED = "failed"


class SideEffect(str, Enum):
    NONE = "none"
    WORKSPACE = "workspace"
    NETWORK = "network"
    PROCESS = "process"
    REGISTRY = "registry"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def utc_timestamp() -> float:
    return time.time()


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {k: jsonable(v) for k, v in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


@dataclass(slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    call_id: str = field(default_factory=lambda: new_id("call"))


@dataclass(slots=True)
class ModelTurn:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    usage: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolResult:
    call_id: str
    tool_name: str
    content: str
    data: dict[str, Any] = field(default_factory=dict)
    is_error: bool = False
    terminal: bool = False
    state_updates: dict[str, Any] = field(default_factory=dict)
    artifact_refs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceRef:
    evidence_id: str
    source_id: str
    locator: str
    quote: str = ""
    title: str = ""
    url: str = ""
    version: str = ""
    retrieved_at: str = ""
    supports: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceRef":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: data[k] for k in allowed if k in data})


@dataclass(slots=True)
class RubricItem:
    criterion: str
    points: float
    required: bool = True
    fatal_if_violated: bool = False
    acceptable_variants: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RubricItem":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: data[k] for k in allowed if k in data})


@dataclass(slots=True)
class TaskBundle:
    task_id: str
    task_type: str
    question: str
    answer: Any
    rationale: str = ""
    rubric: list[RubricItem] = field(default_factory=list)
    evidence: list[EvidenceRef] = field(default_factory=list)
    skill_lineage: list[str] = field(default_factory=list)
    difficulty: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskBundle":
        required = ("task_id", "task_type", "question", "answer")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"TaskBundle missing fields: {', '.join(missing)}")
        return cls(
            task_id=str(data["task_id"]),
            task_type=str(data["task_type"]),
            question=str(data["question"]),
            answer=data["answer"],
            rationale=str(data.get("rationale", "")),
            rubric=[RubricItem.from_dict(x) for x in data.get("rubric", [])],
            evidence=[EvidenceRef.from_dict(x) for x in data.get("evidence", [])],
            skill_lineage=[str(x) for x in data.get("skill_lineage", [])],
            difficulty=dict(data.get("difficulty", {})),
            metadata=dict(data.get("metadata", {})),
            verification=dict(data.get("verification", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


@dataclass(slots=True)
class Event:
    event_type: str
    run_id: str
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: new_id("event"))
    timestamp: float = field(default_factory=utc_timestamp)


@dataclass(slots=True)
class RunState:
    run_id: str
    objective: str
    workspace: str
    stage: Stage = Stage.DISCOVER
    turn: int = 0
    status: str = "running"
    active_skill_ids: list[str] = field(default_factory=list)
    loaded_skill_ids: list[str] = field(default_factory=list)
    enabled_tool_names: list[str] = field(default_factory=list)
    composition_graph: dict[str, Any] = field(default_factory=dict)
    working_artifact: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[dict[str, Any]] = field(default_factory=list)
    failed_checks: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    compact_summary: str = ""
    artifact_refs: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=utc_timestamp)
    updated_at: float = field(default_factory=utc_timestamp)

    @classmethod
    def create(cls, objective: str, workspace: str | Path) -> "RunState":
        return cls(run_id=new_id("run"), objective=objective, workspace=str(Path(workspace)))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunState":
        raw = dict(data)
        raw["stage"] = Stage(raw.get("stage", Stage.DISCOVER.value))
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: raw[k] for k in allowed if k in raw})

    def to_dict(self) -> dict[str, Any]:
        self.updated_at = utc_timestamp()
        return jsonable(self)

    def apply_updates(self, updates: dict[str, Any]) -> None:
        for key, value in updates.items():
            if key == "stage":
                self.stage = Stage(value)
            elif key == "append_failed_check":
                self.failed_checks.append(value)
            elif key == "append_evidence":
                values = value if isinstance(value, list) else [value]
                self.evidence_refs.extend(values)
            elif key == "append_artifact_ref":
                values = value if isinstance(value, list) else [value]
                self.artifact_refs.extend(str(x) for x in values)
            elif key == "enable_tool":
                values = value if isinstance(value, list) else [value]
                for tool_name in values:
                    if tool_name not in self.enabled_tool_names:
                        self.enabled_tool_names.append(str(tool_name))
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                self.metadata[key] = value
        self.updated_at = utc_timestamp()

