from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Iterable

from agentic_proposing.core.models import ModelTurn
from .base import TeacherProvider


class ScriptedTeacher(TeacherProvider):
    """Deterministic provider for tests and offline demonstrations."""

    def __init__(self, turns: Iterable[ModelTurn] = ()):
        self.turns = deque(deepcopy(list(turns)))
        self.requests: list[dict[str, Any]] = []

    def push(self, turn: ModelTurn) -> None:
        self.turns.append(deepcopy(turn))

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> ModelTurn:
        self.requests.append(
            {
                "messages": deepcopy(messages),
                "tools": deepcopy(tools),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self.turns:
            raise RuntimeError("ScriptedTeacher has no remaining turns")
        return self.turns.popleft()

