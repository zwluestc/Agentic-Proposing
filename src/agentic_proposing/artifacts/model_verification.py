from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import json

from agentic_proposing.core.models import TaskBundle
from agentic_proposing.providers.base import TeacherProvider
from agentic_proposing.skills.prompts import render_prompt
from .verification import CheckResult


def _normalized_answer(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.strip().lower().split())
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(slots=True)
class VerifierVote:
    valid: bool
    answer: Any
    ambiguities: list[str] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)
    justification: str = ""
    mode: str = "privileged"

    @classmethod
    def from_dict(cls, data: dict[str, Any], mode: str) -> "VerifierVote":
        return cls(
            valid=bool(data.get("valid", False)),
            answer=data.get("answer"),
            ambiguities=[str(item) for item in data.get("ambiguities", [])],
            fatal_errors=[str(item) for item in data.get("fatal_errors", [])],
            justification=str(data.get("justification", "")),
            mode=mode,
        )


class VerifierCommittee:
    """Independent, isolated SOTA-Teacher calls following the paper's verifier semantics."""

    def __init__(self, provider: TeacherProvider, voters: int = 3, second_audit: bool = True):
        if voters < 1:
            raise ValueError("voters must be positive")
        self.provider = provider
        self.voters = voters
        self.second_audit = second_audit

    def _vote(self, bundle: TaskBundle, context: Any, mode: str) -> VerifierVote:
        task = bundle.to_dict()
        if mode == "blind":
            task.pop("answer", None)
            task.pop("rationale", None)
            task.pop("verification", None)
            source = render_prompt("blind_verifier.md", task=json.dumps(task, ensure_ascii=False, indent=2))
        else:
            events = context.run_store.events.read()[-300:]
            source = (
                "Audit the final TaskBundle with privileged access to its observable proposer trajectory. "
                "Independently solve it, flag ambiguity or underspecification, and return strict JSON with "
                "valid, answer, ambiguities, fatal_errors, justification.\n\n"
                + json.dumps({"task_bundle": task, "trajectory": events}, ensure_ascii=False, indent=2)
            )
        payload = self.provider.complete_json([{"role": "user", "content": source}], temperature=0.1)
        if not isinstance(payload, dict):
            raise ValueError("Verifier must return an object")
        return VerifierVote.from_dict(payload, mode)

    def verify(self, bundle: TaskBundle, context: Any) -> dict[str, Any]:
        votes = [self._vote(bundle, context, "privileged") for _ in range(self.voters)]
        blind = self._vote(bundle, context, "blind")
        majority_threshold = self.voters // 2 + 1
        valid_votes = [vote for vote in votes if vote.valid and not vote.fatal_errors and not vote.ambiguities]
        majority = len(valid_votes) >= majority_threshold
        answer_groups: dict[str, int] = {}
        for vote in valid_votes:
            key = _normalized_answer(vote.answer)
            answer_groups[key] = answer_groups.get(key, 0) + 1
        canonical_answer = max(answer_groups, key=answer_groups.get) if answer_groups else ""
        answer_consistent = bool(answer_groups) and answer_groups[canonical_answer] >= majority_threshold
        blind_is_clear = blind.valid and not blind.fatal_errors and not blind.ambiguities
        blind_matches = bool(blind_is_clear and answer_consistent and _normalized_answer(blind.answer) == canonical_answer)
        audit: dict[str, Any] = {}
        if self.second_audit and majority and not (answer_consistent and blind_matches):
            audit_payload = self.provider.complete_json(
                [
                    {
                        "role": "user",
                        "content": (
                            "Second-audit these independent verifier outputs. Resolve discrepancies only when the task has "
                            "one defensible answer. Return strict JSON: accepted, canonical_answer, objections, justification.\n\n"
                            + json.dumps(
                                {
                                    "task": bundle.to_dict(),
                                    "privileged_votes": [asdict(vote) for vote in votes],
                                    "blind_vote": asdict(blind),
                                },
                                ensure_ascii=False,
                                default=str,
                            )
                        ),
                    }
                ],
                temperature=0.0,
            )
            audit = dict(audit_payload) if isinstance(audit_payload, dict) else {}
        accepted = majority and (
            (answer_consistent and blind_matches)
            or bool(audit.get("accepted", False) and not audit.get("objections"))
        )
        return {
            "accepted": accepted,
            "majority_valid": majority,
            "answer_consistent": answer_consistent,
            "blind_matches": blind_matches,
            "canonical_answer": canonical_answer,
            "votes": [
                {
                    "valid": vote.valid,
                    "answer": vote.answer,
                    "ambiguities": vote.ambiguities,
                    "fatal_errors": vote.fatal_errors,
                    "justification": vote.justification,
                    "mode": vote.mode,
                }
                for vote in votes
            ],
            "blind_vote": {
                "valid": blind.valid,
                "answer": blind.answer,
                "ambiguities": blind.ambiguities,
                "fatal_errors": blind.fatal_errors,
                "justification": blind.justification,
                "mode": blind.mode,
            },
            "second_audit": audit,
        }


class ModelVerifierGate:
    name = "model_verifier_committee"

    def __init__(self, committee: VerifierCommittee):
        self.committee = committee

    def check(self, bundle: TaskBundle, context: Any) -> CheckResult:
        report = self.committee.verify(bundle, context)
        return CheckResult(
            self.name,
            bool(report["accepted"]),
            "fatal",
            "verifier committee accepted" if report["accepted"] else "verifier committee rejected",
            report,
        )

