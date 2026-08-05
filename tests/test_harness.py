from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.config import RuntimeConfig
from agentic_proposing.core.models import ModelTurn, ToolCall
from agentic_proposing.providers.scripted import ScriptedTeacher
from agentic_proposing.runtime import build_runtime


def valid_bundle(task_id: str = "task-1") -> dict:
    return {
        "task_id": task_id,
        "task_type": "calculation",
        "question": "Using the explicitly supplied formula, calculate the unique scalar result for input 4.",
        "answer": 25,
        "rationale": "100 divided by 4 equals 25.",
        "rubric": [],
        "evidence": [],
        "skill_lineage": ["med.calc.two-stage-clearance"],
        "difficulty": {"estimated": 1},
        "metadata": {"knowledge_dependency": False},
    }


class HarnessTests(unittest.TestCase):
    def test_plain_text_does_not_finish_but_submit_does(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            teacher = ScriptedTeacher(
                [
                    ModelTurn(content="I am finished."),
                    ModelTurn(tool_calls=[ToolCall("submit_artifact", {"task_bundle": valid_bundle()})]),
                ]
            )
            runtime = build_runtime(
                temporary,
                teacher,
                RuntimeConfig(harness={"max_turns": 4}, skill_mining={"enabled": False}),
            )
            result = runtime.harness.run("Create one deterministic task")
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.turns, 2)
            events = runtime.layout.run_dir(result.run_id).joinpath("events.jsonl").read_text(encoding="utf-8")
            self.assertIn("non_terminal_text", events)
            self.assertIn("tool_succeeded", events)

    def test_failed_submit_returns_to_refine(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            invalid = valid_bundle("bad")
            invalid["question"] = "short"
            teacher = ScriptedTeacher(
                [
                    ModelTurn(tool_calls=[ToolCall("submit_artifact", {"task_bundle": invalid})]),
                    ModelTurn(tool_calls=[ToolCall("submit_artifact", {"task_bundle": valid_bundle("fixed")})]),
                ]
            )
            runtime = build_runtime(
                temporary,
                teacher,
                RuntimeConfig(harness={"max_turns": 4}, skill_mining={"enabled": False}),
            )
            result = runtime.harness.run("Refine an invalid task")
            self.assertEqual(result.status, "completed")
            self.assertTrue(result.state.failed_checks)
            self.assertEqual(result.state.working_artifact["task_id"], "fixed")

    def test_checkpoints_and_final_artifact_persist(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            teacher = ScriptedTeacher(
                [ModelTurn(tool_calls=[ToolCall("submit_artifact", {"task_bundle": valid_bundle()})])]
            )
            runtime = build_runtime(
                temporary, teacher, RuntimeConfig(skill_mining={"enabled": False})
            )
            result = runtime.harness.run("Persist the run")
            run_dir = runtime.layout.run_dir(result.run_id)
            self.assertTrue((run_dir / "state.json").is_file())
            self.assertTrue(any((run_dir / "checkpoints").glob("*.json")))
            self.assertTrue(any((run_dir / "final").glob("accepted-*.json")))


if __name__ == "__main__":
    unittest.main()

