from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
from contextlib import closing
import json
import sqlite3
import threading

from agentic_proposing.core.models import jsonable
from .models import Document, DocumentChunk


class EvidenceStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS documents_sha ON documents(sha256);
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(document_id),
                    ordinal INTEGER NOT NULL,
                    locator TEXT NOT NULL,
                    section TEXT NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    UNIQUE(document_id, ordinal)
                );
                """
            )
            try:
                connection.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, title, section, text)"
                )
            except sqlite3.OperationalError:
                pass

    def ingest(self, document: Document, chunks: Iterable[DocumentChunk]) -> dict[str, Any]:
        chunk_rows = list(chunks)
        with self._lock, closing(self._connect()) as connection, connection:
            existing = connection.execute(
                "SELECT document_id FROM documents WHERE sha256=?", (document.sha256,)
            ).fetchone()
            if existing:
                return {"document_id": existing["document_id"], "duplicate": True, "chunks": 0}
            connection.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (
                    document.document_id,
                    document.title,
                    document.source_path,
                    document.media_type,
                    document.sha256,
                    json.dumps(document.metadata, ensure_ascii=False, sort_keys=True),
                ),
            )
            fts_available = self._fts_available(connection)
            for chunk in chunk_rows:
                connection.execute(
                    "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.ordinal,
                        chunk.locator,
                        chunk.section,
                        chunk.text,
                        json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                    ),
                )
                if fts_available:
                    connection.execute(
                        "INSERT INTO chunks_fts(chunk_id, title, section, text) VALUES (?, ?, ?, ?)",
                        (chunk.chunk_id, document.title, chunk.section, chunk.text),
                    )
        return {"document_id": document.document_id, "duplicate": False, "chunks": len(chunk_rows)}

    def _fts_available(self, connection: sqlite3.Connection) -> bool:
        return bool(
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
            ).fetchone()
        )

    def search(self, query: str, limit: int = 10, document_ids: list[str] | None = None) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            if self._fts_available(connection):
                sql = """
                    SELECT c.chunk_id, c.document_id, c.locator, c.section, c.text,
                           d.title, d.source_path, bm25(chunks_fts) AS rank
                    FROM chunks_fts
                    JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
                    JOIN documents d ON d.document_id = c.document_id
                    WHERE chunks_fts MATCH ?
                """
                params: list[Any] = [query]
                if document_ids:
                    placeholders = ",".join("?" for _ in document_ids)
                    sql += f" AND c.document_id IN ({placeholders})"
                    params.extend(document_ids)
                sql += " ORDER BY rank LIMIT ?"
                params.append(limit)
                try:
                    rows = connection.execute(sql, params).fetchall()
                except sqlite3.OperationalError:
                    rows = self._like_search(connection, query, limit, document_ids)
            else:
                rows = self._like_search(connection, query, limit, document_ids)
        return [
            {
                "evidence_id": row["chunk_id"],
                "source_id": row["document_id"],
                "locator": row["locator"],
                "section": row["section"],
                "title": row["title"],
                "source_path": row["source_path"],
                "text": row["text"],
                "rank": row["rank"] if "rank" in row.keys() else None,
            }
            for row in rows
        ]

    def _like_search(
        self, connection: sqlite3.Connection, query: str, limit: int, document_ids: list[str] | None
    ) -> list[sqlite3.Row]:
        terms = [item for item in query.split() if item]
        where = " AND ".join("c.text LIKE ?" for _ in terms) or "1=1"
        params: list[Any] = [f"%{item}%" for item in terms]
        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            where += f" AND c.document_id IN ({placeholders})"
            params.extend(document_ids)
        params.append(limit)
        return connection.execute(
            f"""
            SELECT c.chunk_id, c.document_id, c.locator, c.section, c.text,
                   d.title, d.source_path, 0.0 AS rank
            FROM chunks c JOIN documents d ON d.document_id=c.document_id
            WHERE {where} ORDER BY c.document_id, c.ordinal LIMIT ?
            """,
            params,
        ).fetchall()

    def list_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT chunk_id, document_id, ordinal, locator, section, text, metadata_json
                FROM chunks WHERE document_id=? ORDER BY ordinal
                """,
                (document_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def fetch(self, evidence_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT c.*, d.title, d.source_path, d.media_type, d.sha256
                FROM chunks c JOIN documents d ON d.document_id=c.document_id
                WHERE c.chunk_id=?
                """,
                (evidence_id,),
            ).fetchone()
        if not row:
            raise KeyError(evidence_id)
        return dict(row)

    def stats(self) -> dict[str, int]:
        with closing(self._connect()) as connection:
            return {
                "documents": connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0],
                "chunks": connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
            }

