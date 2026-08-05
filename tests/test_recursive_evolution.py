from __future__ import annotations

from tempfile import TemporaryDirectory
import json
import unittest

from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.config import RuntimeConfig
from agentic_proposing.core.models import ModelTurn, ToolCall
from agentic_proposing.evolution import RecursiveEvolutionEngine
from agentic_proposing.providers.scripted import ScriptedTeacher
from agentic_proposing.runtime import build_runtime


def composition_c() -> dict:
    return {
        "skill_id": "med.composite.temporal-choice",
        "name": "Temporal calculation rendered as a choice",
        "version": "0.1.0",
        "description": "Couple a longitudinal deterministic calculation to error-derived choice rendering.",
        "intent": "Require temporal quantitative inference and uniquely gradable option selection.",
        "instances": [
            {"instance_id": "calc", "skill_id": "med.calc.two-stage-clearance"},
            {"instance_id": "choice", "skill_id": "task.choice.numeric-distractors"},
        ],
        "edges": [
            {"from": "calc.case.clearance_delta", "to": "choice.gold.scalar", "data_type": "number"}
        ],
        "inputs": ["case.baseline_value", "case.followup_value"],
        "outputs": ["task.single_choice"],
        "procedure": ["Compute the temporal delta.", "Render the verified delta with error-derived options."],
        "novelty_claim": ["The rendering consumes a two-time-point derived value rather than a supplied scalar."],
    }


def composition_d() -> dict:
    return {
        "skill_id": "med.composite.recursive-contrast",
        "name": "Recursive temporal contrast",
        "version": "0.1.0",
        "description": "Reuse a verified temporal-choice composite and contrast it with a second calculation instance.",
        "intent": "Demonstrate recursive C plus A composition after C passes independent replay.",
        "instances": [
            {"instance_id": "prior", "skill_id": "med.composite.temporal-choice"},
            {"instance_id": "recheck", "skill_id": "med.calc.two-stage-clearance"},
        ],
        "edges": [
            {"from": "recheck.case.clearance_delta", "to": "prior.case.followup_value", "data_type": "number"}
        ],
        "inputs": ["case.series"],
        "outputs": ["task.recursive_contrast"],
        "procedure": ["Instantiate the verified composite.", "Bind its state to a fresh calculation and contrast results."],
        "novelty_claim": ["A promoted composite is recursively reused as a first-class parent."],
    }


def accepted_task(task_id: str, skill_id: str) -> dict:
    return {
        "task_id": task_id,
        "task_type": "single_choice",
        "question": "Under the supplied deterministic construction, which option is the unique verified result?",
        "answer": "4",
        "rationale": "The deterministic replay yields four.",
        "skill_lineage": [skill_id],
        "metadata": {"options": ["3", "4", "5"], "knowledge_dependency": False},
    }


class RecursiveEvolutionTests(unittest.TestCase):
    def test_promoted_c_is_available_as_parent_for_next_round_d(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            teacher = ScriptedTeacher(
                [
                    ModelTurn(content=json.dumps({"compositions": [composition_c()]})),
                    ModelTurn(
                        tool_calls=[
                            ToolCall(
                                "submit_artifact",
                                {"task_bundle": accepted_task("task-c", "med.composite.temporal-choice")},
                            )
                        ]
                    ),
                    ModelTurn(content=json.dumps({"compositions": [composition_d()]})),
                    ModelTurn(
                        tool_calls=[
                            ToolCall(
                                "submit_artifact",
                                {"task_bundle": accepted_task("task-d", "med.composite.recursive-contrast")},
                            )
                        ]
                    ),
                ]
            )
            runtime = build_runtime(
                temporary,
                teacher,
                RuntimeConfig(skill_mining={"enabled": False}),
            )
            reports = RecursiveEvolutionEngine(
                teacher, runtime.registry, runtime.harness
            ).evolve(
                "Build recursive medical reasoning tasks",
                rounds=2,
                compositions_per_round=1,
                replays_per_composition=1,
            )
            self.assertEqual(reports[0].promoted, ["med.composite.temporal-choice"])
            self.assertEqual(reports[1].promoted, ["med.composite.recursive-contrast"])
            self.assertTrue(runtime.registry.has("med.composite.temporal-choice"))
            self.assertTrue(runtime.registry.has("med.composite.recursive-contrast"))
            lineage = runtime.registry.lineage("med.composite.recursive-contrast")
            self.assertIn(
                "med.composite.temporal-choice",
                [parent["skill_id"] for parent in lineage["parents"]],
            )


if __name__ == "__main__":
    unittest.main()
