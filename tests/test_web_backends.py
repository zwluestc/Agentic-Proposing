from __future__ import annotations

from unittest.mock import patch
import json
import unittest

from agentic_proposing.tools.web_backends import SearxngSearchBackend, create_search_backend


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, *args):
        return self.payload


class WebBackendTests(unittest.TestCase):
    def test_searxng_normalizes_results_and_applies_domain_query(self) -> None:
        backend = SearxngSearchBackend({"endpoint": "https://search.example.org"})
        payload = {
            "results": [
                {
                    "title": "Guideline",
                    "url": "https://who.int/example",
                    "content": "Authoritative summary",
                    "publishedDate": "2026-01-01",
                }
            ]
        }
        with patch("agentic_proposing.tools.web_backends.request.urlopen", return_value=FakeResponse(payload)) as call:
            rows = backend.search("renal staging", 5, ["who.int"])
        self.assertEqual(rows[0]["provider"], "searxng")
        self.assertEqual(rows[0]["snippet"], "Authoritative summary")
        requested_url = call.call_args.args[0].full_url
        self.assertIn("site%3Awho.int", requested_url)

    def test_factory_supports_named_backends_and_rejects_unknown(self) -> None:
        self.assertIsNone(create_search_backend({}))
        self.assertIsInstance(
            create_search_backend({"backend": "searxng", "endpoint": "https://search.example.org"}),
            SearxngSearchBackend,
        )
        with self.assertRaises(ValueError):
            create_search_backend({"backend": "unknown"})


if __name__ == "__main__":
    unittest.main()
