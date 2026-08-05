from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from urllib import parse, request
import json
import os

from agentic_proposing.core.models import SideEffect, ToolResult
from .base import BaseTool, ToolContext, ToolRegistry, ToolSpec


def _headers(values: dict[str, Any]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name, value in values.items():
        text = str(value)
        if text.startswith("$ENV:"):
            env_name = text[5:]
            text = os.getenv(env_name, "")
            if not text:
                raise RuntimeError(f"Missing environment variable for external tool header: {env_name}")
        resolved[str(name)] = text
    return resolved


class ExternalHttpTool(BaseTool):
    """A configured JSON-over-HTTP tool exposed through the normal dispatcher and policy layer."""

    def __init__(self, definition: dict[str, Any]):
        self.definition = dict(definition)
        name = str(definition["name"])
        side_effect = SideEffect(str(definition.get("side_effect", "network")))
        self.spec = ToolSpec(
            name=name,
            description=str(definition.get("description", f"External configured tool {name}")),
            input_schema=dict(definition.get("input_schema", {"type": "object"})),
            output_schema=dict(definition.get("output_schema", {})),
            side_effect=side_effect,
            read_only=bool(definition.get("read_only", True)),
            parallelizable=bool(definition.get("parallelizable", True)),
            deferred=bool(definition.get("deferred", True)),
            tags=[str(item) for item in definition.get("tags", ["external", "provider"])],
        )
        self.endpoint = str(definition["endpoint"])
        parsed_endpoint = parse.urlparse(self.endpoint)
        if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.hostname:
            raise ValueError(f"External HTTP tool {name} requires an HTTP(S) endpoint")
        self.method = str(definition.get("method", "POST")).upper()
        if self.method not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError(f"Unsupported external HTTP method: {self.method}")
        self.timeout = int(definition.get("timeout_seconds", 60))
        self.headers = dict(definition.get("headers", {}))

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        headers = {"Accept": "application/json", **_headers(self.headers)}
        endpoint = self.endpoint
        body: bytes | None = None
        if self.method == "GET":
            encoded = parse.urlencode({key: value for key, value in arguments.items()}, doseq=True)
            endpoint += ("&" if "?" in endpoint else "?") + encoded
        else:
            body = json.dumps(arguments, ensure_ascii=False).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        req = request.Request(endpoint, data=body, headers=headers, method=self.method)
        with request.urlopen(req, timeout=self.timeout) as response:
            raw = response.read(10_000_001)
            content_type = response.headers.get("Content-Type", "")
            status = getattr(response, "status", 200)
        if len(raw) > 10_000_000:
            raise ValueError("External tool response exceeds 10 MB")
        text = raw.decode("utf-8", errors="replace")
        if "json" in content_type.lower():
            data: Any = json.loads(text)
        else:
            data = {"text": text}
        return ToolResult(
            call_id,
            self.spec.name,
            f"External tool completed with HTTP {status}",
            data={"response": data, "status": status},
        )


class ToolProvider(Protocol):
    provider_id: str

    def load_tools(self) -> list[BaseTool]: ...


@dataclass(slots=True)
class ConfiguredToolProvider:
    provider_id: str
    config: dict[str, Any]

    def load_tools(self) -> list[BaseTool]:
        definitions = list(self.config.get("tools", []))
        catalog_url = str(self.config.get("catalog_url", ""))
        if catalog_url:
            parsed_catalog = parse.urlparse(catalog_url)
            if parsed_catalog.scheme not in {"http", "https"} or not parsed_catalog.hostname:
                raise ValueError("External tool catalog requires an HTTP(S) URL")
            req = request.Request(
                catalog_url,
                headers={"Accept": "application/json", **_headers(dict(self.config.get("headers", {})))},
            )
            with request.urlopen(req, timeout=int(self.config.get("timeout_seconds", 30))) as response:
                raw = response.read(5_000_001)
            if len(raw) > 5_000_000:
                raise ValueError("External tool catalog exceeds 5 MB")
            payload = json.loads(raw.decode("utf-8"))
            remote = payload.get("tools", payload)
            if not isinstance(remote, list):
                raise ValueError("External tool catalog must be a list or {tools: list}")
            definitions.extend(remote)
        return [ExternalHttpTool(dict(item)) for item in definitions]


class ToolProviderManager:
    def __init__(self, registry: ToolRegistry, providers: list[ToolProvider]):
        self.registry = registry
        self.providers = providers
        self._owned: dict[str, set[str]] = {}
        self._protected = set(registry.names())

    def refresh(self) -> dict[str, Any]:
        loaded_by_provider: dict[str, list[BaseTool]] = {}
        owner_by_name: dict[str, str] = {}
        for provider in self.providers:
            loaded = provider.load_tools()
            names = [tool.spec.name for tool in loaded]
            if len(names) != len(set(names)):
                raise ValueError(f"Provider {provider.provider_id} returned duplicate tool names")
            conflicts = sorted(name for name in names if name in self._protected)
            if conflicts:
                raise ValueError(f"External tools cannot override built-ins: {conflicts}")
            cross_provider = sorted(name for name in names if name in owner_by_name)
            if cross_provider:
                raise ValueError(
                    f"External tool names must be globally unique; {cross_provider} already owned by another provider"
                )
            for name in names:
                owner_by_name[name] = provider.provider_id
            loaded_by_provider[provider.provider_id] = loaded

        next_owned = {
            provider_id: {tool.spec.name for tool in loaded}
            for provider_id, loaded in loaded_by_provider.items()
        }
        all_previous = set().union(*self._owned.values()) if self._owned else set()
        all_next = set().union(*next_owned.values()) if next_owned else set()
        for stale in all_previous - all_next:
            self.registry.unregister(stale)
        for loaded in loaded_by_provider.values():
            for tool in loaded:
                self.registry.upsert(tool)

        report: dict[str, Any] = {"providers": {}}
        for provider_id, names in next_owned.items():
            previous = self._owned.get(provider_id, set())
            report["providers"][provider_id] = {
                "loaded": sorted(names),
                "removed": sorted(previous - names),
            }
        self._owned = next_owned
        report["tool_count"] = len(self.registry.names())
        return report



class ToolCatalogRefreshTool(BaseTool):
    spec = ToolSpec(
        name="tool_catalog_refresh",
        description="Refresh configured external tool providers and update the discoverable tool catalog at runtime.",
        input_schema={"type": "object", "properties": {}},
        side_effect=SideEffect.NETWORK,
        read_only=False,
        deferred=False,
        tags=["tools", "catalog", "refresh", "external", "manage"],
    )

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        report = context.services["tool_provider_manager"].refresh()
        return ToolResult(call_id, self.spec.name, "External tool catalog refreshed", data=report)


def providers_from_config(config: dict[str, Any]) -> list[ConfiguredToolProvider]:
    providers: list[ConfiguredToolProvider] = []
    for index, item in enumerate(config.get("providers", [])):
        value = dict(item)
        providers.append(ConfiguredToolProvider(str(value.get("id", f"provider-{index + 1}")), value))
    return providers
