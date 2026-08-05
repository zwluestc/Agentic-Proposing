from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.config import RuntimeConfig
from agentic_proposing.core.models import ModelTurn, ToolCall
from agentic_proposing.providers.scripted import ScriptedTeacher
from agentic_proposing.runtime import build_runtime


class PostAcceptTests(unittest.TestCase):
    def test_skill_mining_failure_does_not_reject_accepted_task(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            bundle = {
                "task_id": "accepted-even-if-mining-fails",
                "task_type": "calculation",
                "question": "Using the supplied deterministic relation, calculate the unique scalar output for input four.",
                "answer": 25,
                "rationale": "One hundred divided by four is twenty-five.",
                "skill_lineage": ["med.calc.two-stage-clearance"],
                "metadata": {"knowledge_dependency": False},
            }
            teacher = ScriptedTeacher(
                [ModelTurn(tool_calls=[ToolCall("submit_artifact", {"task_bundle": bundle})])]
            )
            runtime = build_runtime(
                temporary,
                teacher,
                RuntimeConfig(skill_mining={"enabled": True}),
            )
            result = runtime.harness.run("Accept the task and attempt optional reverse mining")
            self.assertEqual(result.status, "completed")
            mining = result.terminal_result.data["skill_mining"]
            self.assertIn("error", mining)
            events = runtime.layout.run_dir(result.run_id).joinpath("events.jsonl").read_text(encoding="utf-8")
            self.assertIn("post_accept_skill_mining_failed", events)


if __name__ == "__main__":
    unittest.main()
