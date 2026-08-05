from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import hashlib

from agentic_proposing.core.models import SideEffect, ToolResult
from agentic_proposing.corpus.chunking import SemanticChunker
from agentic_proposing.corpus.models import Document
from .base import BaseTool, ToolContext, ToolSpec
from .web_tools import fetch_public_page


class EvidenceIngestUrlTool(BaseTool):
    spec = ToolSpec(
        name="evidence_ingest_url",
        description=(
            "Fetch one public allowlisted page, preserve URL/retrieval/hash provenance, chunk it, and ingest it into "
            "the local evidence store. Use evidence_search afterwards to select exact supporting chunks."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "title": {"type": "string"},
                "max_bytes": {"type": "integer", "minimum": 1000, "maximum": 5000000},
                "target_chars": {"type": "integer", "minimum": 500, "maximum": 10000},
                "overlap_chars": {"type": "integer", "minimum": 0, "maximum": 2000},
            },
            "required": ["url"],
        },
        side_effect=SideEffect.NETWORK,
        read_only=False,
        deferred=True,
        tags=["web", "evidence", "ingest", "reference", "provenance"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        fetched = fetch_public_page(
            str(arguments["url"]),
            [str(item) for item in context.services.get("web_domain_allowlist", [])],
            min(int(arguments.get("max_bytes", 2_000_000)), 5_000_000),
            int(context.services.get("web_fetch_timeout", 30)),
        )
        text = str(fetched["text"])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        document_id = f"web_{digest[:20]}"
        retrieved_at = datetime.now(timezone.utc).isoformat()
        document = Document(
            document_id=document_id,
            title=str(arguments.get("title") or fetched.get("title") or fetched["final_url"]),
            source_path=str(fetched["final_url"]),
            media_type=str(fetched.get("content_type", "text/html")),
            text=text,
            sha256=digest,
            metadata={
                "requested_url": fetched["requested_url"],
                "final_url": fetched["final_url"],
                "retrieved_at": retrieved_at,
                "sha256": digest,
                "provenance": "web_fetch",
            },
        )
        chunks = SemanticChunker(
            int(arguments.get("target_chars", 2600)),
            int(arguments.get("overlap_chars", 300)),
        ).chunk(document)
        store = context.services["evidence_store"]
        outcome = store.ingest(document, chunks)
        stored_chunks = store.list_document_chunks(str(outcome["document_id"]))
        manifest = {
            "document": {
                "document_id": document_id,
                "title": document.title,
                "url": fetched["final_url"],
                "retrieved_at": retrieved_at,
                "sha256": digest,
            },
            "ingest": outcome,
            "chunks": [
                {
                    "evidence_id": row["chunk_id"],
                    "locator": row["locator"],
                    "section": row["section"],
                }
                for row in stored_chunks
            ],
        }
        ref = context.run_store.write_artifact(f"web-evidence-{document_id}.json", manifest, "artifacts")
        evidence_refs = [
            {
                "evidence_id": row["chunk_id"],
                "source_id": str(outcome["document_id"]),
                "locator": row["locator"],
                "title": document.title,
                "url": str(fetched["final_url"]),
                "retrieved_at": retrieved_at,
            }
            for row in stored_chunks
        ]
        return ToolResult(
            call_id,
            self.spec.name,
            f"Ingested web page as {len(stored_chunks)} evidence chunks",
            data={**manifest, "artifact_ref": ref},
            state_updates={"append_evidence": evidence_refs},
            artifact_refs=[ref],
        )
