from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import os
import shutil
import tempfile

from .models import Event, RunState, jsonable


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


@dataclass(frozen=True, slots=True)
class WorkspaceLayout:
    root: Path

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def skills(self) -> Path:
        return self.root / "skills"

    @property
    def evidence(self) -> Path:
        return self.root / "evidence"

    @property
    def sources(self) -> Path:
        return self.root / "sources"

    @property
    def staging(self) -> Path:
        return self.root / "staging"

    def run_dir(self, run_id: str) -> Path:
        return self.runs / run_id

    def initialize(self) -> None:
        for path in (
            self.runs,
            self.skills / "atomic",
            self.skills / "composite",
            self.skills / "deprecated",
            self.evidence / "documents",
            self.evidence / "chunks",
            self.sources,
            self.staging / "skills",
            self.staging / "tasks",
        ):
            path.mkdir(parents=True, exist_ok=True)


class EventStore:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "events.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Event) -> None:
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(jsonable(event), ensure_ascii=False, sort_keys=True) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def tail(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.read()[-limit:]


class RunStore:
    def __init__(self, layout: WorkspaceLayout, run_id: str):
        self.layout = layout
        self.run_dir = layout.run_dir(run_id)
        self.state_path = self.run_dir / "state.json"
        self.artifact_dir = self.run_dir / "artifacts"
        self.tool_output_dir = self.run_dir / "tool_outputs"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        for path in (self.artifact_dir, self.tool_output_dir, self.checkpoint_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.events = EventStore(self.run_dir)

    def save_state(self, state: RunState) -> None:
        atomic_write_text(
            self.state_path,
            json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        )

    def load_state(self) -> RunState:
        return RunState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))

    def write_artifact(self, name: str, value: Any, subdir: str = "artifacts") -> str:
        base = self.run_dir / subdir
        base.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c for c in name if c.isalnum() or c in "-_.") or "artifact.json"
        path = base / safe_name
        if isinstance(value, str):
            content = value
        else:
            content = json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        atomic_write_text(path, content)
        return str(path.relative_to(self.layout.root)).replace("\\", "/")

    def checkpoint(self, state: RunState, label: str = "turn") -> str:
        filename = f"{state.turn:04d}-{label}.json"
        path = self.checkpoint_dir / filename
        atomic_write_text(path, json.dumps(state.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return str(path.relative_to(self.layout.root)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy_source(source: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / source.name
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return destination

