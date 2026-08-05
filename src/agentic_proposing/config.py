from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

from .providers.base import TeacherProvider
from .providers.anthropic import AnthropicTeacher
from .providers.openai_compatible import OpenAICompatibleTeacher


def load_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("YAML config requires the optional yaml dependency") from exc
        value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError("Configuration root must be an object")
    return value


@dataclass(slots=True)
class RuntimeConfig:
    provider: dict[str, Any] = field(default_factory=dict)
    harness: dict[str, Any] = field(default_factory=dict)
    permissions: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    python: dict[str, Any] = field(default_factory=dict)
    web: dict[str, Any] = field(default_factory=dict)
    external_tools: dict[str, Any] = field(default_factory=dict)
    skill_mining: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "RuntimeConfig":
        raw = load_mapping(path)
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: dict(raw.get(key, {})) for key in allowed})


def create_provider(config: dict[str, Any]) -> TeacherProvider:
    kind = str(config.get("type", "openai_compatible"))
    if kind == "openai_compatible":
        return OpenAICompatibleTeacher(
            base_url=str(config.get("base_url", "https://api.openai.com/v1")),
            model=str(config["model"]),
            api_key=config.get("api_key"),
            api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
            timeout_seconds=int(config.get("timeout_seconds", 180)),
            extra_headers=dict(config.get("extra_headers", {})),
        )
    if kind == "anthropic":
        return AnthropicTeacher(
            model=str(config["model"]),
            api_key=config.get("api_key"),
            api_key_env=str(config.get("api_key_env", "ANTHROPIC_API_KEY")),
            base_url=str(config.get("base_url", "https://api.anthropic.com")),
            timeout_seconds=int(config.get("timeout_seconds", 180)),
        )
    raise ValueError(f"Unknown provider type: {kind}")

