from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import unittest

from agentic_proposing.artifacts.model_verification import VerifierCommittee
from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.config import RuntimeConfig
from agentic_proposing.core.models import ModelTurn, TaskBundle
from agentic_proposing.providers.scripted import ScriptedTeacher
from agentic_proposing.runtime import build_runtime


def vote(answer: str, *, valid: bool = True) -> ModelTurn:
    return ModelTurn(
        content=json.dumps(
            {
                "valid": valid,
                "answer": answer,
                "ambiguities": [],
                "fatal_errors": [],
                "justification": "independent solve",
            }
        )
    )


def bundle() -> TaskBundle:
    return TaskBundle.from_dict(
        {
            "task_id": "verifier-1",
            "task_type": "single_choice",
            "question": "Which supplied option equals two plus two under ordinary integer arithmetic?",
            "answer": "4",
            "metadata": {"options": ["3", "4", "5"], "knowledge_dependency": False},
        }
    )


def context() -> SimpleNamespace:
    events = SimpleNamespace(read=lambda: [{"event_type": "candidate_created"}])
    return SimpleNamespace(run_store=SimpleNamespace(events=events))


class VerifierCommitteeTests(unittest.TestCase):
    def test_blind_vote_must_match_privileged_majority_not_minority(self) -> None:
        teacher = ScriptedTeacher([vote("4"), vote("4"), vote("5"), vote("5")])
        report = VerifierCommittee(teacher, voters=3, second_audit=False).verify(bundle(), context())
        self.assertTrue(report["answer_consistent"])
        self.assertEqual(report["canonical_answer"], "4")
        self.assertFalse(report["blind_matches"])
        self.assertFalse(report["accepted"])

    def test_second_audit_serializes_slot_votes_and_can_resolve_disagreement(self) -> None:
        audit = ModelTurn(
            content=json.dumps(
                {
                    "accepted": True,
                    "canonical_answer": "4",
                    "objections": [],
                    "justification": "The majority answer follows from the task.",
                }
            )
        )
        teacher = ScriptedTeacher([vote("4"), vote("4"), vote("5"), vote("5"), audit])
        report = VerifierCommittee(teacher, voters=3, second_audit=True).verify(bundle(), context())
        self.assertTrue(report["accepted"])
        self.assertTrue(report["second_audit"]["accepted"])
        self.assertEqual(len(teacher.requests), 5)

    def test_runtime_exposes_management_and_optional_verifier_tools(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            runtime = build_runtime(
                temporary,
                ScriptedTeacher(),
                RuntimeConfig(
                    verification={"model_committee_enabled": True},
                    skill_mining={"enabled": False},
                ),
            )
            names = set(runtime.tools.names())
            self.assertTrue(
                {
                    "skill_prune",
                    "skill_promote",
                    "skill_deprecate",
                    "skill_version",
                    "verifier_committee",
                }.issubset(names)
            )


if __name__ == "__main__":
    unittest.main()
