from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
import json

from agentic_proposing.core.harness import AgentHarness, HarnessResult
from agentic_proposing.core.models import RunState
from agentic_proposing.providers.base import TeacherProvider
from agentic_proposing.skills.extraction import ExtractionOutcome, SkillExtractor
from agentic_proposing.skills.models import CompositeSpec
from agentic_proposing.skills.prompts import render_prompt
from agentic_proposing.skills.registry import SkillRegistry


@dataclass(slots=True)
class EvolutionRound:
    round_index: int
    composed: list[str] = field(default_factory=list)
    proposed_runs: list[str] = field(default_factory=list)
    promoted: list[str] = field(default_factory=list)
    staged_from_tasks: list[str] = field(default_factory=list)
    validations: dict[str, dict[str, Any]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


class RecursiveEvolutionEngine:
    """No-RL recursive closure over skill composition, task synthesis, and reverse skill mining."""

    def __init__(
        self,
        provider: TeacherProvider,
        registry: SkillRegistry,
        harness: AgentHarness,
    ):
        self.provider = provider
        self.registry = registry
        self.harness = harness
        self.extractor = SkillExtractor(provider, registry)

    def acquire_qa(self, items: Iterable[dict[str, Any]]) -> list[ExtractionOutcome]:
        return [self.extractor.extract_from_qa(item) for item in items]

    def acquire_corpus(self, chunks: Iterable[dict[str, Any]]) -> list[ExtractionOutcome]:
        return [self.extractor.extract_from_corpus(chunk) for chunk in chunks]

    def propose_compositions(self, goal: str, limit: int = 4) -> list[CompositeSpec]:
        metadata = self.registry.discovery_summaries()
        prompt = render_prompt(
            "composition.md",
            goal=goal,
            skills=json.dumps(metadata, ensure_ascii=False, indent=2),
            limit=str(limit),
        )
        payload = self.provider.complete_json([{"role": "user", "content": prompt}])
        rows = payload if isinstance(payload, list) else payload.get("compositions", [])
        return [CompositeSpec.from_dict(row) for row in rows[:limit]]

    def evolve(
        self,
        goal: str,
        rounds: int = 2,
        compositions_per_round: int = 3,
        replays_per_composition: int = 2,
        min_replay_success_rate: float = 0.8,
        auto_promote: bool = True,
    ) -> list[EvolutionRound]:
        if rounds < 1 or compositions_per_round < 1 or replays_per_composition < 1:
            raise ValueError("rounds, compositions_per_round, and replays_per_composition must be positive")
        if not 0.0 <= min_replay_success_rate <= 1.0:
            raise ValueError("min_replay_success_rate must be between 0 and 1")
        reports: list[EvolutionRound] = []
        for index in range(1, rounds + 1):
            report = EvolutionRound(index)
            try:
                specs = self.propose_compositions(goal, compositions_per_round)
            except Exception as exc:
                report.errors.append(f"composition proposal: {type(exc).__name__}: {exc}")
                reports.append(report)
                continue
            for spec in specs:
                try:
                    staged_path = self.registry.stage_composite(spec)
                    report.composed.append(spec.skill_id)
                except Exception as exc:
                    report.errors.append(f"{spec.skill_id}: stage: {type(exc).__name__}: {exc}")
                    continue
                successful = 0
                task_ids: list[str] = []
                run_ids: list[str] = []
                for replay_index in range(1, replays_per_composition + 1):
                    try:
                        state = RunState.create(
                            (
                                f"Instantiate and verify composite skill {spec.skill_id} for goal: {goal}. "
                                f"Independent replay {replay_index}/{replays_per_composition}."
                            ),
                            self.registry.layout.root,
                        )
                        state.stage = __import__(
                            "agentic_proposing.core.models", fromlist=["Stage"]
                        ).Stage.COMPOSE
                        state.active_skill_ids = [node.skill_id for node in spec.instances]
                        state.loaded_skill_ids = list(dict.fromkeys(state.active_skill_ids))
                        state.composition_graph = {
                            "candidate_skill_id": spec.skill_id,
                            "replay_index": replay_index,
                            "instances": [
                                {
                                    "instance_id": node.instance_id,
                                    "skill_id": node.skill_id,
                                    "bind": node.bind,
                                }
                                for node in spec.instances
                            ],
                        }
                        result = self.harness.run(state.objective, state)
                        report.proposed_runs.append(result.run_id)
                        run_ids.append(result.run_id)
                        if result.status == "completed" and result.terminal_result is not None:
                            successful += 1
                            task_id = str(result.state.working_artifact.get("task_id", ""))
                            if task_id:
                                task_ids.append(task_id)
                            mining = result.terminal_result.data.get("skill_mining", {})
                            report.staged_from_tasks.extend(mining.get("staged", []))
                    except Exception as exc:
                        report.errors.append(
                            f"{spec.skill_id}: replay {replay_index}: {type(exc).__name__}: {exc}"
                        )
                success_rate = successful / replays_per_composition
                validation = {
                    "passed": successful > 0 and success_rate >= min_replay_success_rate,
                    "validation_type": "independent_task_instantiation_replay",
                    "replay_total": replays_per_composition,
                    "replay_successful": successful,
                    "replay_success_rate": success_rate,
                    "min_replay_success_rate": min_replay_success_rate,
                    "run_ids": run_ids,
                    "task_ids": task_ids,
                    "round_index": index,
                }
                report.validations[spec.skill_id] = validation
                if auto_promote and validation["passed"]:
                    try:
                        self.registry.promote(staged_path.name, validation)
                        report.promoted.append(spec.skill_id)
                    except Exception as exc:
                        report.errors.append(
                            f"{spec.skill_id}: promote: {type(exc).__name__}: {exc}"
                        )
            self.registry.refresh()
            reports.append(report)
        return reports

