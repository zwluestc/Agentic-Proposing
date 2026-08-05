from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.skills.compiler import CompositionError, SkillCompiler
from agentic_proposing.skills.models import CompositeSpec
from agentic_proposing.skills.registry import SkillRegistry


def repeated_spec(cycle: bool = False) -> CompositeSpec:
    edges = [
        {"from": "time1.case.baseline_clearance", "to": "time2.case.baseline_value", "data_type": "number"}
    ]
    if cycle:
        edges.append({"from": "time2.case.followup_clearance", "to": "time1.case.baseline_value"})
    return CompositeSpec.from_dict(
        {
            "skill_id": "med.calc.repeated-recursive",
            "name": "Repeated recursive calculation",
            "version": "0.1.0",
            "description": "Apply the same calculation skill twice with changed bindings.",
            "intent": "Model a state transition across repeated operation instances.",
            "instances": [
                {"instance_id": "time1", "skill_id": "med.calc.two-stage-clearance", "bind": {"case.baseline_value": "case.t0"}},
                {"instance_id": "time2", "skill_id": "med.calc.two-stage-clearance", "bind": {"case.baseline_value": "case.t1"}},
            ],
            "edges": edges,
            "inputs": ["case.t0", "case.t1"],
            "outputs": ["case.transition"],
            "procedure": ["Calculate at time1.", "Bind the changed state to time2.", "Compare outputs."],
            "novelty_claim": ["Same skill type is applied at distinct state bindings."],
        }
    )


class SkillTests(unittest.TestCase):
    def test_discovery_does_not_load_body(self) -> None:
        with TemporaryDirectory() as temporary:
            layout = bootstrap_demo_workspace(temporary)
            registry = SkillRegistry(layout)
            package = registry.get_package("med.calc.two-stage-clearance")
            self.assertIsNone(package._body)
            registry.discovery_summaries()
            self.assertIsNone(package._body)
            loaded = registry.load_for_context("med.calc.two-stage-clearance", 10000)
            self.assertIn("## Action", loaded)
            self.assertIsNotNone(package._body)

    def test_same_skill_can_repeat_with_distinct_bindings(self) -> None:
        with TemporaryDirectory() as temporary:
            layout = bootstrap_demo_workspace(temporary)
            registry = SkillRegistry(layout)
            destination = registry.stage_composite(repeated_spec())
            self.assertTrue((destination / "composition.yaml").is_file())
            self.assertTrue((destination / "lineage.json").is_file())

    def test_cycle_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            layout = bootstrap_demo_workspace(temporary)
            registry = SkillRegistry(layout)
            with self.assertRaises(CompositionError):
                registry.stage_composite(repeated_spec(cycle=True))

    def test_duplicate_instance_id_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            layout = bootstrap_demo_workspace(temporary)
            registry = SkillRegistry(layout)
            spec = repeated_spec()
            spec.instances[1].instance_id = "time1"
            errors = SkillCompiler(registry).validate(spec)
            self.assertTrue(any("instance_id" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

