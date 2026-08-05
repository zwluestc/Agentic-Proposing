from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from agentic_proposing.core.models import SideEffect, ToolResult, jsonable
from agentic_proposing.corpus.chunking import SemanticChunker
from agentic_proposing.corpus.parsers import BenchmarkParser, DocumentParser
from .base import BaseTool, ToolContext, ToolSpec


def _allowed_source(path: Path, context: ToolContext) -> Path:
    resolved = path.resolve()
    roots = [context.layout.root.resolve()]
    roots.extend(Path(item).resolve() for item in context.services.get("source_roots", []))
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise PermissionError(f"Source is outside configured roots: {resolved}")
    return resolved


class CorpusIngestTool(BaseTool):
    spec = ToolSpec(
        name="corpus_ingest",
        description="Parse a textbook, paper, HTML, EPUB, DOCX, Markdown, text, or PDF into provenance-rich evidence chunks.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "title": {"type": "string"},
                "target_chars": {"type": "integer", "minimum": 500, "maximum": 10000},
                "overlap_chars": {"type": "integer", "minimum": 0, "maximum": 2000},
            },
            "required": ["path"],
        },
        side_effect=SideEffect.WORKSPACE,
        read_only=False,
        deferred=True,
        tags=["textbook", "ebook", "corpus", "pdf", "epub", "evidence", "parse"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        path = _allowed_source(Path(str(arguments["path"])), context)
        document = DocumentParser().parse(path, {"title": arguments.get("title") or path.stem})
        chunker = SemanticChunker(
            target_chars=int(arguments.get("target_chars", 2600)),
            overlap_chars=int(arguments.get("overlap_chars", 300)),
        )
        chunks = chunker.chunk(document)
        store = context.services["evidence_store"]
        outcome = store.ingest(document, chunks)
        manifest_ref = context.run_store.write_artifact(
            f"ingest-{document.document_id}.json",
            {
                "document": {
                    "document_id": document.document_id,
                    "title": document.title,
                    "source_path": document.source_path,
                    "media_type": document.media_type,
                    "sha256": document.sha256,
                    "metadata": document.metadata,
                },
                "outcome": outcome,
            },
            "tool_outputs",
        )
        return ToolResult(
            call_id,
            self.spec.name,
            f"Ingested {document.title}: {outcome['chunks']} chunks",
            data=outcome,
            artifact_refs=[manifest_ref],
        )


class BenchmarkParseTool(BaseTool):
    spec = ToolSpec(
        name="benchmark_parse",
        description="Parse JSON/JSONL/CSV/TSV benchmark rows into normalized question-answer items with row provenance.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
        },
        side_effect=SideEffect.WORKSPACE,
        read_only=False,
        deferred=True,
        tags=["benchmark", "dataset", "jsonl", "extract", "parse"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        path = _allowed_source(Path(str(arguments["path"])), context)
        items = BenchmarkParser().parse(path)
        if arguments.get("limit"):
            items = items[: int(arguments["limit"])]
        ref = context.run_store.write_artifact(
            f"benchmark-{path.stem}.json",
            [jsonable(item) for item in items],
            "tool_outputs",
        )
        preview = [
            {"item_id": item.item_id, "task_type": item.task_type, "question": item.question[:300]}
            for item in items[:5]
        ]
        return ToolResult(
            call_id,
            self.spec.name,
            f"Parsed {len(items)} benchmark items",
            data={"count": len(items), "preview": preview, "artifact_ref": ref},
            artifact_refs=[ref],
        )


class EvidenceSearchTool(BaseTool):
    spec = ToolSpec(
        name="evidence_search",
        description="Search the local trusted Evidence Store and return exact source locators and bounded text spans.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 30},
                "document_ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
        read_only=True,
        parallelizable=True,
        deferred=True,
        tags=["evidence", "search", "reference", "grounding"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        store = context.services["evidence_store"]
        rows = store.search(
            str(arguments["query"]),
            int(arguments.get("limit", 8)),
            [str(x) for x in arguments.get("document_ids", [])] or None,
        )
        bounded = [{**row, "text": row["text"][:4000]} for row in rows]
        evidence = [
            {
                "evidence_id": row["evidence_id"],
                "source_id": row["source_id"],
                "locator": row["locator"],
                "title": row["title"],
            }
            for row in rows
        ]
        return ToolResult(
            call_id,
            self.spec.name,
            f"Found {len(rows)} evidence chunks",
            data={"results": bounded},
            state_updates={"append_evidence": evidence} if evidence else {},
        )


class EvidenceFetchTool(BaseTool):
    spec = ToolSpec(
        name="evidence_fetch",
        description="Fetch one exact evidence chunk by ID for citation or verifier inspection.",
        input_schema={
            "type": "object",
            "properties": {"evidence_id": {"type": "string"}},
            "required": ["evidence_id"],
        },
        read_only=True,
        parallelizable=True,
        deferred=True,
        tags=["evidence", "fetch", "citation"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        row = context.services["evidence_store"].fetch(str(arguments["evidence_id"]))
        row["text"] = row["text"][:12000]
        return ToolResult(call_id, self.spec.name, "Evidence fetched", data=row)

