from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .models import SkillManifest
from .registry import SkillRegistry


@dataclass(slots=True)
class ReplayCase:
    case_id: str
    bindings: dict[str, Any]
    source_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReplayReport:
    passed: bool
    total: int
    successful: int
    distinct_sources: int
    details: list[dict[str, Any]]

    def to_validation(self, min_success_rate: float = 0.8, min_sources: int = 2) -> dict[str, Any]:
        rate = self.successful / max(1, self.total)
        return {
            "passed": self.passed and rate >= min_success_rate and self.distinct_sources >= min_sources,
            "replay_total": self.total,
            "replay_successful": self.successful,
            "replay_success_rate": rate,
            "distinct_sources": self.distinct_sources,
            "details": self.details,
        }


class ReplayHarness:
    """Tests transfer across bindings/sources before a staged skill may be promoted."""

    def run(
        self,
        manifest: SkillManifest,
        cases: list[ReplayCase],
        evaluator: Callable[[SkillManifest, ReplayCase], dict[str, Any]],
    ) -> ReplayReport:
        details: list[dict[str, Any]] = []
        successful = 0
        sources: set[str] = set()
        for case in cases:
            result = evaluator(manifest, case)
            passed = bool(result.get("passed", False))
            successful += int(passed)
            sources.update(case.source_ids)
            details.append({"case_id": case.case_id, "passed": passed, **result})
        return ReplayReport(bool(cases) and successful > 0, len(cases), successful, len(sources), details)

