from __future__ import annotations

from types import SimpleNamespace
import unittest

from agentic_proposing.tools.web_tools import WebFetchTool


class WebSafetyTests(unittest.TestCase):
    def test_fetch_blocks_loopback_and_private_literals_before_request(self) -> None:
        context = SimpleNamespace(services={"web_domain_allowlist": []})
        tool = WebFetchTool()
        for url in ("http://127.0.0.1/secret", "http://10.0.0.8/", "http://[::1]/"):
            with self.subTest(url=url), self.assertRaises(PermissionError):
                tool.invoke({"url": url}, context, "call-test")


if __name__ == "__main__":
    unittest.main()
