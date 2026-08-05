from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree
import csv
import hashlib
import io
import json
import re
import zipfile

from agentic_proposing.core.models import new_id
from .models import BenchmarkItem, Document


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav"}:
            self._ignored += 1
        elif tag in {"p", "div", "h1", "h2", "h3", "h4", "li", "tr", "br"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav"} and self._ignored:
            self._ignored -= 1
        elif tag in {"p", "div", "h1", "h2", "h3", "h4", "li", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts)
        value = re.sub(r"[ \t]+", " ", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "utf-16"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


class DocumentParser:
    SUPPORTED = {".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".epub", ".docx"}

    def parse(self, path: str | Path, metadata: dict[str, Any] | None = None) -> Document:
        source = Path(path).resolve()
        suffix = source.suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported document type: {suffix}")
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if suffix in {".txt", ".md", ".markdown"}:
            text, media_type = _decode_bytes(raw), "text/markdown" if suffix != ".txt" else "text/plain"
        elif suffix in {".html", ".htm"}:
            parser = _TextExtractor()
            parser.feed(_decode_bytes(raw))
            text, media_type = parser.text(), "text/html"
        elif suffix == ".pdf":
            text, media_type = self._pdf(source), "application/pdf"
        elif suffix == ".epub":
            text, media_type = self._epub(source), "application/epub+zip"
        else:
            text, media_type = self._docx(source), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        title = str((metadata or {}).get("title") or source.stem)
        return Document(new_id("doc"), title, str(source), media_type, text, digest, dict(metadata or {}))

    def _pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PDF parsing requires `pip install .[pdf]`") from exc
        pages = []
        for index, page in enumerate(PdfReader(str(path)).pages, 1):
            pages.append(f"\n\n[Page {index}]\n" + (page.extract_text() or ""))
        return "".join(pages).strip()

    def _epub(self, path: Path) -> str:
        parts: list[str] = []
        with zipfile.ZipFile(path) as archive:
            names = sorted(
                name for name in archive.namelist() if name.lower().endswith((".xhtml", ".html", ".htm"))
            )
            for name in names:
                parser = _TextExtractor()
                parser.feed(_decode_bytes(archive.read(name)))
                value = parser.text()
                if value:
                    parts.append(f"\n\n[EPUB:{name}]\n{value}")
        return "".join(parts).strip()

    def _docx(self, path: Path) -> str:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        parts: list[str] = []
        for element in root.iter():
            if element.tag.endswith("}t") and element.text:
                parts.append(element.text)
            elif element.tag.endswith("}p"):
                parts.append("\n")
        return re.sub(r"\n{3,}", "\n\n", "".join(parts)).strip()


class BenchmarkParser:
    QUESTION_KEYS = ("question", "prompt", "problem", "instruction", "input")
    ANSWER_KEYS = ("answer", "gold", "label", "output", "target")
    TYPE_KEYS = ("task_type", "type", "category")
    OPTION_KEYS = ("options", "choices", "candidates")

    def parse(self, path: str | Path) -> list[BenchmarkItem]:
        source = Path(path).resolve()
        suffix = source.suffix.lower()
        if suffix == ".jsonl":
            rows = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        elif suffix == ".json":
            value = json.loads(source.read_text(encoding="utf-8-sig"))
            rows = value if isinstance(value, list) else value.get("items", value.get("data", []))
        elif suffix in {".csv", ".tsv"}:
            with source.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle, delimiter="\t" if suffix == ".tsv" else ","))
        else:
            raise ValueError(f"Unsupported benchmark format: {suffix}")
        if not isinstance(rows, list):
            raise ValueError("Benchmark root must be a list or contain items/data list")
        return [self._map_row(row, source, index) for index, row in enumerate(rows, 1)]

    def _pick(self, row: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
        for key in keys:
            if key in row:
                return row[key]
        return default

    def _map_row(self, row: dict[str, Any], source: Path, index: int) -> BenchmarkItem:
        if not isinstance(row, dict):
            raise ValueError(f"Benchmark row {index} is not an object")
        question = self._pick(row, self.QUESTION_KEYS)
        if question is None:
            raise ValueError(f"Benchmark row {index} has no recognized question field")
        answer = self._pick(row, self.ANSWER_KEYS)
        options = self._pick(row, self.OPTION_KEYS, []) or []
        known = set(self.QUESTION_KEYS + self.ANSWER_KEYS + self.TYPE_KEYS + self.OPTION_KEYS)
        return BenchmarkItem(
            item_id=str(row.get("id") or f"{source.stem}:{index}"),
            question=str(question),
            answer=answer,
            task_type=str(self._pick(row, self.TYPE_KEYS, "unknown")),
            options=list(options) if isinstance(options, list) else [options],
            rubric=row.get("rubric"),
            source_path=str(source),
            row_number=index,
            metadata={key: value for key, value in row.items() if key not in known},
        )

