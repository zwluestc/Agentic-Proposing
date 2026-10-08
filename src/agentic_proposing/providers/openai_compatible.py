from __future__ import annotations

from typing import Any
from urllib import error, request
import json
import os
import ssl
import time

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
        extra_body: dict[str, Any] | None = None,
        ca_bundle: str | None = None,
        max_retries: int = 2,
        max_output_tokens: int | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get(api_key_env, "")
        self.timeout_seconds = timeout_seconds
        self.extra_headers = dict(extra_headers or {})
        self.extra_body = dict(extra_body or {})
        self.ssl_context = ssl.create_default_context(cafile=ca_bundle) if ca_bundle else None
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens

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
            "max_tokens": min(max_tokens, self.max_output_tokens) if self.max_output_tokens else max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        payload.update(self.extra_body)
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        for attempt in range(self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.timeout_seconds, context=self.ssl_context) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except error.HTTPError as exc:
                if exc.code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(min(8, 2 ** attempt))
                    continue
                detail = exc.read().decode("utf-8", errors="replace")
                if self.api_key:
                    detail = detail.replace(self.api_key, "[REDACTED]")
                raise RuntimeError(f"Teacher HTTP {exc.code}: {detail[:2000]}") from exc
            except (error.URLError, TimeoutError):
                if attempt >= self.max_retries:
                    raise
                time.sleep(min(8, 2 ** attempt))
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
