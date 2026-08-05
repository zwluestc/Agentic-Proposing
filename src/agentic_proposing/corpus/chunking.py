from __future__ import annotations

from dataclasses import dataclass
import re

from agentic_proposing.core.models import new_id
from .models import Document, DocumentChunk


HEADING_RE = re.compile(
    r"^(?:#{1,6}\s+.+|\u7b2c[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341\u767e\u5343\u4e070-9]+[\u7ae0\u8282\u7bc7\u90e8]\s*.*|[A-Z0-9]+(?:\.[0-9]+)*[.\u3001]\s*.+)$"
)


@dataclass(slots=True)
class SemanticChunker:
    target_chars: int = 2600
    overlap_chars: int = 300
    min_chars: int = 180

    def _sections(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        heading = ""
        buffer: list[str] = []
        for line in text.replace("\r\n", "\n").split("\n"):
            stripped = line.strip()
            if stripped and HEADING_RE.match(stripped) and buffer:
                sections.append((heading, "\n".join(buffer).strip()))
                heading, buffer = stripped, []
            else:
                buffer.append(line)
        if buffer:
            sections.append((heading, "\n".join(buffer).strip()))
        return [(heading, body) for heading, body in sections if body]

    def chunk(self, document: Document) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        ordinal = 0
        for section, body in self._sections(document.text):
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
            buffer = ""
            for paragraph in paragraphs:
                candidate = f"{buffer}\n\n{paragraph}".strip() if buffer else paragraph
                if len(candidate) <= self.target_chars or len(buffer) < self.min_chars:
                    buffer = candidate
                    continue
                chunks.append(self._make_chunk(document, ordinal, section, buffer))
                ordinal += 1
                overlap = buffer[-self.overlap_chars :] if self.overlap_chars else ""
                buffer = f"{overlap}\n\n{paragraph}".strip()
                while len(buffer) > self.target_chars * 2:
                    split_at = self._split_point(buffer, self.target_chars)
                    piece, buffer = buffer[:split_at], buffer[split_at:]
                    chunks.append(self._make_chunk(document, ordinal, section, piece.strip()))
                    ordinal += 1
            if buffer:
                chunks.append(self._make_chunk(document, ordinal, section, buffer))
                ordinal += 1
        return chunks

    def _split_point(self, text: str, target: int) -> int:
        window = text[max(1, target - 300) : target + 300]
        matches = list(re.finditer(r"[\u3002\uff01\uff1f!?]\s*", window))
        if matches:
            return max(1, target - 300 + matches[-1].end())
        return target

    def _make_chunk(self, document: Document, ordinal: int, section: str, text: str) -> DocumentChunk:
        locator = f"section={section or 'root'};chunk={ordinal}"
        return DocumentChunk(
            chunk_id=new_id("chunk"),
            document_id=document.document_id,
            ordinal=ordinal,
            locator=locator,
            text=text,
            section=section,
            metadata={"source_sha256": document.sha256},
        )

