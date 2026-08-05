from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.core.models import ModelTurn
from agentic_proposing.providers.scripted import ScriptedTeacher
from agentic_proposing.skills.extraction import SkillExtractor, SkillMiner
from agentic_proposing.skills.registry import SkillRegistry
from agentic_proposing.core.models import TaskBundle


def candidate_payload() -> dict:
    return {
        "skills": [
            {
                "manifest": {
                    "skill_id": "task.boundary.unique-row",
                    "name": "Unique decision row",
                    "version": "0.1.0",
                    "kind": "atomic",
                    "status": "candidate",
                    "description": "Construct a decision-table case that binds exactly one row at a boundary.",
                    "intent": "Test explicit boundary handling without ambiguity.",
                    "triggers": ["decision table", "boundary"],
                    "negative_triggers": [],
                    "inputs": ["table.rows", "case.values"],
                    "outputs": ["decision.unique_row"],
                    "difficulty_effect": {"base": 3},
                    "tools": [],
                    "evidence_contract": {},
                    "composition": {},
                    "source_refs": [],
                    "metrics": {},
                    "tags": ["boundary", "decision-table"]
                },
                "body": "# Skill\n\n## Structure\nTable.\n## Action\nBind one row.\n## Effect\nUnique output.",
                "lineage": {"reasoning_nodes": ["boundary selection"]},
                "tests": {"positive": [{"case": "one match"}], "negative": [{"case": "two matches"}]}
            }
        ]
    }


class ExtractionTests(unittest.TestCase):
    def test_question_level_extraction_stages_with_source_lineage(self) -> None:
        with TemporaryDirectory() as temporary:
            layout = bootstrap_demo_workspace(temporary)
            registry = SkillRegistry(layout)
            teacher = ScriptedTeacher([ModelTurn(content=json.dumps(candidate_payload()))])
            outcome = SkillExtractor(teacher, registry).extract_from_qa(
                {"item_id": "q1", "question": "Choose the unique boundary row", "answer": "A"}
            )
            self.assertEqual(outcome.staged, ["task.boundary.unique-row"])
            staged = layout.staging / "skills" / "task.boundary.unique-row@0.1.0"
            lineage = json.loads((staged / "lineage.json").read_text(encoding="utf-8"))
            self.assertEqual(lineage["extracted_from"]["type"], "benchmark_item")

    def test_task_mining_records_task_and_run(self) -> None:
        with TemporaryDirectory() as temporary:
            layout = bootstrap_demo_workspace(temporary)
            registry = SkillRegistry(layout)
            teacher = ScriptedTeacher([ModelTurn(content=json.dumps(candidate_payload()))])
            bundle = TaskBundle.from_dict(
                {
                    "task_id": "accepted-1",
                    "task_type": "calculation",
                    "question": "Calculate a unique result from the supplied deterministic formula.",
                    "answer": 1,
                    "metadata": {"knowledge_dependency": False},
                }
            )
            outcome = SkillMiner(teacher, registry).mine(bundle, [{"event_type": "tool_succeeded"}], "run-1")
            self.assertEqual(outcome.staged, ["task.boundary.unique-row"])
            lineage = json.loads(
                (layout.staging / "skills" / "task.boundary.unique-row@0.1.0" / "lineage.json").read_text(encoding="utf-8")
            )
            self.assertEqual(lineage["extracted_from"]["run_id"], "run-1")
            self.assertEqual(lineage["extracted_from"]["task_id"], "accepted-1")


if __name__ == "__main__":
    unittest.main()

