from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Document:
    document_id: str
    title: str
    source_path: str
    media_type: str
    text: str
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    ordinal: int
    locator: str
    text: str
    section: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BenchmarkItem:
    item_id: str
    question: str
    answer: Any
    task_type: str = "unknown"
    options: list[Any] = field(default_factory=list)
    rubric: Any = None
    source_path: str = ""
    row_number: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

