from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from agentic_proposing.corpus.chunking import SemanticChunker
from agentic_proposing.corpus.parsers import BenchmarkParser, DocumentParser
from agentic_proposing.corpus.store import EvidenceStore


class CorpusTests(unittest.TestCase):
    def test_document_chunk_ingest_search_and_fetch(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "book.md"
            source.write_text(
                "# Rule\n\nA boundary value must select exactly one row.\n\n# Failure\n\nOverlapping rows are ambiguous.",
                encoding="utf-8",
            )
            document = DocumentParser().parse(source)
            chunks = SemanticChunker(target_chars=80, overlap_chars=0, min_chars=10).chunk(document)
            store = EvidenceStore(root / "evidence.sqlite")
            outcome = store.ingest(document, chunks)
            self.assertGreater(outcome["chunks"], 0)
            results = store.search("boundary", 5)
            self.assertTrue(results)
            fetched = store.fetch(results[0]["evidence_id"])
            self.assertEqual(fetched["document_id"], document.document_id)
            self.assertIn("locator", fetched)

    def test_benchmark_parser_preserves_row_provenance(self) -> None:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "bench.jsonl"
            source.write_text(
                json.dumps({"id": "q1", "question": "What is 2+2?", "answer": 4}) + "\n",
                encoding="utf-8",
            )
            item = BenchmarkParser().parse(source)[0]
            self.assertEqual(item.item_id, "q1")
            self.assertEqual(item.row_number, 1)
            self.assertEqual(item.source_path, str(source.resolve()))


if __name__ == "__main__":
    unittest.main()

