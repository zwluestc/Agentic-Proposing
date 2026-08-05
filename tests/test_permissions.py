from __future__ import annotations

import unittest

from agentic_proposing.core.hooks import PermissionPolicy


class PermissionPolicyTests(unittest.TestCase):
    def test_read_only_does_not_bypass_network_or_process_policy(self) -> None:
        policy = PermissionPolicy(
            allow_read_only=True,
            allow_network=False,
            allow_process=False,
        )
        self.assertFalse(policy.decide("web_fetch", "network", True).allow)
        self.assertFalse(policy.decide("skill_python", "process", True).allow)
        self.assertTrue(policy.decide("evidence_search", "none", True).allow)

    def test_read_only_can_be_disabled_and_explicit_allow_is_narrow_override(self) -> None:
        policy = PermissionPolicy(
            allowed={"web_search"},
            denied={"blocked_search"},
            allow_read_only=False,
            allow_network=False,
        )
        self.assertFalse(policy.decide("evidence_search", "none", True).allow)
        self.assertTrue(policy.decide("web_search", "network", True).allow)
        self.assertFalse(policy.decide("blocked_search", "network", True).allow)


if __name__ == "__main__":
    unittest.main()
