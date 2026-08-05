from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import json
import re

from agentic_proposing.core.models import ModelTurn


class TeacherProvider(ABC):
    """Model adapter used by the proposer, composer, extractor, and judges."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> ModelTurn:
        raise NotImplementedError

    def complete_json(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> dict[str, Any] | list[Any]:
        turn = self.complete(messages, [], temperature=temperature, max_tokens=max_tokens)
        return parse_json_payload(turn.content)


def parse_json_payload(text: str) -> dict[str, Any] | list[Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
                return value
            except json.JSONDecodeError:
                pass
        raise ValueError("Teacher response did not contain valid JSON")

