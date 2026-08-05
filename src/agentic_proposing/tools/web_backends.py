from __future__ import annotations

from typing import Any
from urllib import parse, request
import json
import os

from .web_tools import SearchBackend


def _json_request(req: request.Request, timeout: int) -> Any:
    with request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw.decode("utf-8"))


def _api_key(config: dict[str, Any], default_env: str) -> str:
    value = config.get("api_key") or os.getenv(str(config.get("api_key_env", default_env)), "")
    if not value:
        raise RuntimeError(f"Missing API key; set {config.get('api_key_env', default_env)}")
    return str(value)


class BraveSearchBackend(SearchBackend):
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.endpoint = str(config.get("endpoint", "https://api.search.brave.com/res/v1/web/search"))
        self.timeout = int(config.get("timeout_seconds", 30))

    def search(self, query: str, limit: int, allowed_domains: list[str]) -> list[dict[str, Any]]:
        if allowed_domains:
            query = f"{query} " + " OR ".join(f"site:{domain}" for domain in allowed_domains)
        params = parse.urlencode({"q": query, "count": min(limit, 20), "safesearch": "moderate"})
        req = request.Request(
            f"{self.endpoint}?{params}",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": _api_key(self.config, "BRAVE_SEARCH_API_KEY"),
            },
        )
        payload = _json_request(req, self.timeout)
        rows = payload.get("web", {}).get("results", [])
        return [
            {
                "title": str(row.get("title", "")),
                "url": str(row.get("url", "")),
                "snippet": str(row.get("description", "")),
                "published": str(row.get("age", row.get("page_age", ""))),
                "provider": "brave",
            }
            for row in rows[:limit]
        ]


class TavilySearchBackend(SearchBackend):
    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.endpoint = str(config.get("endpoint", "https://api.tavily.com/search"))
        self.timeout = int(config.get("timeout_seconds", 30))

    def search(self, query: str, limit: int, allowed_domains: list[str]) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "query": query,
            "max_results": min(limit, 20),
            "search_depth": str(self.config.get("search_depth", "advanced")),
            "include_answer": False,
            "include_raw_content": False,
        }
        if allowed_domains:
            body["include_domains"] = allowed_domains
        req = request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {_api_key(self.config, 'TAVILY_API_KEY')}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        payload = _json_request(req, self.timeout)
        return [
            {
                "title": str(row.get("title", "")),
                "url": str(row.get("url", "")),
                "snippet": str(row.get("content", "")),
                "score": row.get("score"),
                "published": str(row.get("published_date", "")),
                "provider": "tavily",
            }
            for row in payload.get("results", [])[:limit]
        ]


class SearxngSearchBackend(SearchBackend):
    def __init__(self, config: dict[str, Any]):
        endpoint = str(config.get("endpoint", "")).rstrip("/")
        if not endpoint:
            raise ValueError("SearXNG backend requires web.endpoint")
        self.endpoint = endpoint
        self.timeout = int(config.get("timeout_seconds", 30))
        self.language = str(config.get("language", "all"))

    def search(self, query: str, limit: int, allowed_domains: list[str]) -> list[dict[str, Any]]:
        if allowed_domains:
            query = f"{query} " + " OR ".join(f"site:{domain}" for domain in allowed_domains)
        params = parse.urlencode({"q": query, "format": "json", "language": self.language})
        req = request.Request(f"{self.endpoint}/search?{params}", headers={"Accept": "application/json"})
        payload = _json_request(req, self.timeout)
        return [
            {
                "title": str(row.get("title", "")),
                "url": str(row.get("url", "")),
                "snippet": str(row.get("content", "")),
                "published": str(row.get("publishedDate", "")),
                "provider": "searxng",
            }
            for row in payload.get("results", [])[:limit]
        ]


def create_search_backend(config: dict[str, Any]) -> SearchBackend | None:
    kind = str(config.get("backend", "")).strip().lower()
    if not kind:
        return None
    if kind == "brave":
        return BraveSearchBackend(config)
    if kind == "tavily":
        return TavilySearchBackend(config)
    if kind == "searxng":
        return SearxngSearchBackend(config)
    raise ValueError(f"Unknown web search backend: {kind}")
