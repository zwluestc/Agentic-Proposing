from __future__ import annotations

import unittest

from agentic_proposing.cli import build_parser


class CliTests(unittest.TestCase):
    def test_skills_workspace_option_is_accepted_after_nested_command(self) -> None:
        parser = build_parser()
        listed = parser.parse_args(["skills", "list", "--workspace", "workspace"])
        versioned = parser.parse_args(
            [
                "skills",
                "version",
                "skill.id",
                "1.1.0",
                "--change-summary",
                "tighten checks",
                "--workspace",
                "workspace",
            ]
        )
        self.assertEqual(listed.workspace, "workspace")
        self.assertEqual(versioned.skills_command, "version")


if __name__ == "__main__":
    unittest.main()
