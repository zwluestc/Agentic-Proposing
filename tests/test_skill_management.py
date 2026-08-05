from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.skills.management import deprecate_skill, fork_skill_version
from agentic_proposing.skills.registry import SkillRegistry


class SkillManagementTests(unittest.TestCase):
    def test_version_replacement_archives_prior_active_package(self) -> None:
        with TemporaryDirectory() as temporary:
            layout = bootstrap_demo_workspace(temporary)
            registry = SkillRegistry(layout)
            skill_id = "med.calc.two-stage-clearance"
            staged = fork_skill_version(registry, skill_id, "1.1.0", "Tighten unit checks")
            destination = registry.promote(
                staged.name,
                {"passed": True, "replay_pass_rate": 1.0},
                replace_existing=True,
            )
            self.assertEqual(registry.get_manifest(skill_id).version, "1.1.0")
            self.assertEqual(registry.get_manifest(skill_id).status, "verified")
            self.assertTrue((layout.skills / "deprecated" / f"{skill_id}@1.0.0").is_dir())
            self.assertEqual(destination, layout.skills / "atomic" / skill_id)

    def test_replacement_requires_explicit_version_lineage(self) -> None:
        with TemporaryDirectory() as temporary:
            layout = bootstrap_demo_workspace(temporary)
            registry = SkillRegistry(layout)
            skill_id = "task.choice.numeric-distractors"
            package = registry.get_package(skill_id)
            staged = registry.stage_atomic(
                package.manifest,
                package.body,
                {"derivation": "invalid replacement"},
            )
            with self.assertRaises(ValueError):
                registry.promote(
                    staged.name,
                    {"passed": True, "allow_duplicate": True},
                    replace_existing=True,
                )

    def test_deprecation_preserves_package_outside_discovery(self) -> None:
        with TemporaryDirectory() as temporary:
            layout = bootstrap_demo_workspace(temporary)
            registry = SkillRegistry(layout)
            skill_id = "task.choice.numeric-distractors"
            destination = deprecate_skill(registry, skill_id, "superseded")
            self.assertFalse(registry.has(skill_id))
            self.assertTrue((destination / "SKILL.md").is_file())
            self.assertEqual(destination.name, f"{skill_id}@1.0.0")


if __name__ == "__main__":
    unittest.main()
