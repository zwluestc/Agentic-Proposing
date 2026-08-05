from __future__ import annotations

from abc import ABC, abstractmethod
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib import parse, request
import ipaddress
import json
import re
import socket

from agentic_proposing.core.models import SideEffect, ToolResult
from .base import BaseTool, ToolContext, ToolSpec


class SearchBackend(ABC):
    @abstractmethod
    def search(self, query: str, limit: int, allowed_domains: list[str]) -> list[dict[str, Any]]:
        raise NotImplementedError


class DisabledSearchBackend(SearchBackend):
    def search(self, query: str, limit: int, allowed_domains: list[str]) -> list[dict[str, Any]]:
        raise RuntimeError("No web search backend configured; use local evidence_search or configure SearchBackend")


class HttpJsonSearchBackend(SearchBackend):
    """Adapter for a caller-provided JSON search endpoint; avoids coupling to one vendor."""

    def __init__(self, url: str, headers: dict[str, str] | None = None, timeout: int = 30):
        self.url = url
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.timeout = timeout

    def search(self, query: str, limit: int, allowed_domains: list[str]) -> list[dict[str, Any]]:
        body = json.dumps(
            {"query": query, "limit": limit, "allowed_domains": allowed_domains}
        ).encode("utf-8")
        req = request.Request(self.url, data=body, headers=self.headers, method="POST")
        with request.urlopen(req, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload.get("results", payload)
        if not isinstance(rows, list):
            raise ValueError("Search endpoint must return a list or {results: list}")
        return rows[:limit]


class WebSearchTool(BaseTool):
    spec = ToolSpec(
        name="web_search",
        description="Search the web through a configured backend, optionally restricted to authoritative domains. Results are pointers, not evidence until fetched.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "allowed_domains": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["query"],
        },
        side_effect=SideEffect.NETWORK,
        read_only=True,
        parallelizable=True,
        deferred=True,
        tags=["web", "search", "reference", "current-information"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        backend = context.services.get("web_search_backend") or DisabledSearchBackend()
        domains = [
            str(x).strip().lower().lstrip(".")
            for x in arguments.get("allowed_domains", [])
            if str(x).strip().lstrip(".")
        ]
        rows = backend.search(str(arguments["query"]), int(arguments.get("limit", 8)), domains)
        if domains:
            rows = [
                row for row in rows
                if any(
                    (lambda hostname: hostname == domain or hostname.endswith("." + domain))(
                        (parse.urlparse(str(row.get("url", ""))).hostname or "").lower()
                    )
                    for domain in domains
                )
            ]
        return ToolResult(call_id, self.spec.name, f"Found {len(rows)} web results", data={"results": rows})


class _HTMLText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.ignore = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "nav"}:
            self.ignore += 1
        elif tag in {"p", "div", "h1", "h2", "h3", "li", "br", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav"} and self.ignore:
            self.ignore -= 1

    def handle_data(self, data: str) -> None:
        if not self.ignore:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self.parts)).strip()


def _require_public_hostname(hostname: str) -> None:
    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(row[4][0])
                for row in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
            }
        except socket.gaierror as exc:
            raise ValueError(f"Hostname could not be resolved: {hostname}") from exc
    if not addresses or any(not address.is_global for address in addresses):
        raise PermissionError(f"Private, local, or reserved network target is blocked: {hostname}")


def fetch_public_page(
    url: str,
    allowlist: list[str],
    max_bytes: int,
    timeout: int = 30,
) -> dict[str, Any]:
    parsed = parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP(S) URLs are supported")
    domains = [str(item).lower().lstrip(".") for item in allowlist if str(item).strip()]

    def validate_target(target_url: str) -> str:
        target = parse.urlparse(target_url)
        if target.scheme not in {"http", "https"} or not target.hostname:
            raise ValueError("Redirect target is not public HTTP(S)")
        hostname = target.hostname.lower()
        _require_public_hostname(hostname)
        if domains and not any(hostname == domain or hostname.endswith("." + domain) for domain in domains):
            raise PermissionError(f"Domain is not allowlisted: {hostname}")
        return hostname

    validate_target(url)
    req = request.Request(url, headers={"User-Agent": "Agentic-Proposing-Repro/0.2"})
    with request.urlopen(req, timeout=timeout) as response:
        final_url = response.geturl()
        validate_target(final_url)
        raw = response.read(max_bytes + 1)
        content_type = response.headers.get("Content-Type", "")
    if len(raw) > max_bytes:
        raise ValueError("Response exceeds max_bytes")
    charset = "utf-8"
    match = re.search(r"charset=([^;\s]+)", content_type, re.IGNORECASE)
    if match:
        charset = match.group(1).strip('"\'')
    try:
        decoded = raw.decode(charset, errors="replace")
    except LookupError:
        decoded = raw.decode("utf-8", errors="replace")
    title = ""
    if "html" in content_type.lower() or "<html" in decoded[:1000].lower():
        title_match = re.search(r"<title[^>]*>(.*?)</title>", decoded, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = re.sub(r"\s+", " ", unescape(title_match.group(1))).strip()
        parser = _HTMLText()
        parser.feed(decoded)
        decoded = parser.text()
    return {
        "requested_url": url,
        "final_url": final_url,
        "content_type": content_type,
        "title": title,
        "text": decoded,
    }


class WebFetchTool(BaseTool):
    spec = ToolSpec(
        name="web_fetch",
        description="Fetch a bounded public HTTP(S) page from an allowlisted domain and return normalized text plus retrieval metadata.",
        input_schema={
            "type": "object",
            "properties": {"url": {"type": "string"}, "max_bytes": {"type": "integer"}},
            "required": ["url"],
        },
        side_effect=SideEffect.NETWORK,
        read_only=True,
        parallelizable=True,
        deferred=True,
        tags=["web", "fetch", "reference", "citation"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        page = fetch_public_page(
            str(arguments["url"]),
            [str(item) for item in context.services.get("web_domain_allowlist", [])],
            min(int(arguments.get("max_bytes", 2_000_000)), 5_000_000),
            int(context.services.get("web_fetch_timeout", 30)),
        )
        ref = context.run_store.write_artifact(
            f"web-{abs(hash(page['final_url']))}.txt", page["text"][:200_000], "tool_outputs"
        )
        return ToolResult(
            call_id,
            self.spec.name,
            "Web page fetched",
            data={**page, "text": page["text"][:12000], "artifact_ref": ref},
            artifact_refs=[ref],
        )
