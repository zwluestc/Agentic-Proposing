from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
import json

from agentic_proposing.core.models import Event, TaskBundle, jsonable
from agentic_proposing.providers.base import TeacherProvider
from .models import SkillManifest
from .prompts import render_prompt
from .registry import SkillRegistry


@dataclass(slots=True)
class ExtractionOutcome:
    staged: list[str]
    rejected: list[dict[str, Any]]


class SkillExtractor:
    """Reverse-engineers Q&A or corpus chunks into staged atomic skill packages."""

    def __init__(self, provider: TeacherProvider, registry: SkillRegistry):
        self.provider = provider
        self.registry = registry

    def extract_from_qa(self, item: dict[str, Any]) -> ExtractionOutcome:
        prompt = render_prompt("skill_extraction_qa.md", source=json.dumps(item, ensure_ascii=False, indent=2))
        payload = self.provider.complete_json([{"role": "user", "content": prompt}])
        return self._stage_payload(payload, {"type": "benchmark_item", "source": item})

    def extract_from_corpus(self, chunk: dict[str, Any]) -> ExtractionOutcome:
        prompt = render_prompt(
            "skill_extraction_corpus.md", source=json.dumps(chunk, ensure_ascii=False, indent=2)
        )
        payload = self.provider.complete_json([{"role": "user", "content": prompt}])
        return self._stage_payload(payload, {"type": "corpus_chunk", "source": chunk})

    def _stage_payload(self, payload: dict[str, Any] | list[Any], source: dict[str, Any]) -> ExtractionOutcome:
        candidates = payload if isinstance(payload, list) else payload.get("skills", [])
        staged: list[str] = []
        rejected: list[dict[str, Any]] = []
        for candidate in candidates:
            try:
                manifest = SkillManifest.from_dict(dict(candidate["manifest"]))
                body = str(candidate["body"])
                lineage = {**dict(candidate.get("lineage", {})), "extracted_from": source}
                duplicates = self.registry.duplicate_report(manifest)
                if any(item["relation"] == "duplicate_of" for item in duplicates):
                    rejected.append(
                        {"skill_id": manifest.skill_id, "reason": "duplicate", "matches": duplicates}
                    )
                    continue
                self.registry.stage_atomic(manifest, body, lineage, dict(candidate.get("tests", {})))
                staged.append(manifest.skill_id)
            except Exception as exc:
                rejected.append({"candidate": candidate, "reason": f"{type(exc).__name__}: {exc}"})
        return ExtractionOutcome(staged, rejected)


class SkillMiner:
    """Extracts emergent skills from an accepted task and full observable trajectory."""

    def __init__(self, provider: TeacherProvider, registry: SkillRegistry, max_events: int = 300):
        self.provider = provider
        self.registry = registry
        self.max_events = max_events

    def mine(self, bundle: TaskBundle, events: list[dict[str, Any]], run_id: str) -> ExtractionOutcome:
        source = {
            "task_bundle": bundle.to_dict(),
            "trajectory": events[-self.max_events :],
            "run_id": run_id,
        }
        prompt = render_prompt("skill_mining.md", source=json.dumps(source, ensure_ascii=False, indent=2))
        payload = self.provider.complete_json([{"role": "user", "content": prompt}])
        extractor = SkillExtractor(self.provider, self.registry)
        return extractor._stage_payload(
            payload,
            {
                "type": "accepted_task_and_trajectory",
                "task_id": bundle.task_id,
                "run_id": run_id,
                "skill_lineage": bundle.skill_lineage,
            },
        )


class PostAcceptSkillMining:
    def __init__(self, miner: SkillMiner):
        self.miner = miner

    def run(self, bundle: TaskBundle, tool_context: Any) -> dict[str, Any]:
        outcome = self.miner.mine(
            bundle, tool_context.run_store.events.read(), tool_context.state.run_id
        )
        tool_context.run_store.events.append(
            Event(
                "post_accept_skill_mining",
                tool_context.state.run_id,
                {"staged": outcome.staged, "rejected": outcome.rejected},
            )
        )
        return {"staged": outcome.staged, "rejected": outcome.rejected}

