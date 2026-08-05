from __future__ import annotations

from typing import Any
from urllib import error, request
import json
import os

from agentic_proposing.core.models import ModelTurn, ToolCall
from .base import TeacherProvider


class AnthropicTeacher(TeacherProvider):
    """Dependency-free adapter for the Anthropic Messages API."""

    def __init__(self, *, model: str, api_key: str | None = None, api_key_env: str = "ANTHROPIC_API_KEY", base_url: str = "https://api.anthropic.com", anthropic_version: str = "2023-06-01", timeout_seconds: int = 180):
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.base_url = base_url.rstrip("/")
        self.anthropic_version = anthropic_version
        self.timeout_seconds = timeout_seconds

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], *, temperature: float = 0.2, max_tokens: int = 8192) -> ModelTurn:
        if not self.api_key:
            raise RuntimeError("Anthropic API key is missing")
        system_parts: list[str] = []
        api_messages: list[dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                system_parts.append(str(message.get("content", "")))
            elif role == "tool":
                api_messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": message.get("tool_call_id", ""), "content": str(message.get("content", ""))}]})
            elif role == "assistant" and message.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": str(message["content"])})
                for call in message["tool_calls"]:
                    function = call.get("function", {})
                    arguments = function.get("arguments", {})
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments or "{}")
                    blocks.append({"type": "tool_use", "id": call.get("id", ""), "name": function.get("name", ""), "input": arguments})
                api_messages.append({"role": "assistant", "content": blocks})
            else:
                api_messages.append({"role": role, "content": str(message.get("content", ""))})
        api_tools = [{"name": item["function"]["name"], "description": item["function"].get("description", ""), "input_schema": item["function"].get("parameters", {"type": "object"})} for item in tools]
        payload: dict[str, Any] = {"model": self.model, "system": "\n\n".join(system_parts), "messages": api_messages, "temperature": temperature, "max_tokens": max_tokens}
        if api_tools:
            payload["tools"] = api_tools
        req = request.Request(f"{self.base_url}/v1/messages", data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json", "x-api-key": self.api_key, "anthropic-version": self.anthropic_version}, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Anthropic HTTP {exc.code}: {detail[:2000]}") from exc
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in body.get("content", []):
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                calls.append(ToolCall(name=block.get("name", ""), arguments=dict(block.get("input", {})), call_id=block.get("id", "")))
        return ModelTurn(content="\n".join(texts), tool_calls=calls, raw=body, usage=dict(body.get("usage", {})))

