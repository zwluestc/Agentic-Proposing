from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import hashlib
import re

from .models import SkillManifest


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def token_set(text: str) -> set[str]:
    normalized = normalize_text(text)
    words = set(normalized.split())
    chars = {normalized[i : i + 3] for i in range(max(0, len(normalized) - 2))}
    return words | chars


def jaccard(left: str, right: str) -> float:
    a, b = token_set(left), token_set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


@dataclass(slots=True)
class DuplicateMatch:
    skill_id: str
    score: float
    relation: str


def find_duplicates(
    candidate: SkillManifest,
    existing: Iterable[SkillManifest],
    threshold: float = 0.82,
) -> list[DuplicateMatch]:
    candidate_text = " ".join(
        [candidate.description, candidate.intent, *candidate.inputs, *candidate.outputs, *candidate.triggers]
    )
    matches: list[DuplicateMatch] = []
    for item in existing:
        item_text = " ".join([item.description, item.intent, *item.inputs, *item.outputs, *item.triggers])
        score = jaccard(candidate_text, item_text)
        if score >= threshold:
            relation = "duplicate_of" if score >= 0.94 else "overlaps"
            matches.append(DuplicateMatch(item.skill_id, score, relation))
    return sorted(matches, key=lambda match: -match.score)

