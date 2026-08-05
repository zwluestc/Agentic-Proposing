from __future__ import annotations

from pathlib import Path


def load_prompt(name: str) -> str:
    path = Path(__file__).resolve().parent.parent / "prompts" / name
    return path.read_text(encoding="utf-8")


def render_prompt(name: str, **values: str) -> str:
    text = load_prompt(name)
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    return text

