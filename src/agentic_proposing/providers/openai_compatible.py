from __future__ import annotations

from typing import Any
from urllib import error, request
import json
import os

from agentic_proposing.core.models import ModelTurn, ToolCall
from .base import TeacherProvider


class OpenAICompatibleTeacher(TeacherProvider):
    """Small dependency-free adapter for OpenAI-compatible chat-completions APIs."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: int = 180,
        extra_headers: dict[str, str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.timeout_seconds = timeout_seconds
        self.extra_headers = dict(extra_headers or {})

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> ModelTurn:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Teacher HTTP {exc.code}: {detail[:2000]}") from exc
        message = body["choices"][0]["message"]
        calls: list[ToolCall] = []
        for raw_call in message.get("tool_calls", []) or []:
            function = raw_call.get("function", {})
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments or "{}")
            calls.append(
                ToolCall(
                    name=function.get("name", ""),
                    arguments=dict(arguments),
                    call_id=raw_call.get("id") or "",
                )
            )
        return ModelTurn(
            content=message.get("content") or "",
            tool_calls=calls,
            raw=body,
            usage=dict(body.get("usage", {})),
        )

