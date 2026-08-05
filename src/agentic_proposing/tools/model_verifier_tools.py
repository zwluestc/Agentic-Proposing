from __future__ import annotations

from typing import Any

from agentic_proposing.artifacts.model_verification import VerifierCommittee
from agentic_proposing.core.models import SideEffect, TaskBundle, ToolResult
from .base import BaseTool, ToolContext, ToolSpec


class VerifierCommitteeTool(BaseTool):
    spec = ToolSpec(
        name="verifier_committee",
        description="Run isolated privileged verifier votes plus a blind solver vote on a candidate TaskBundle before submission.",
        input_schema={
            "type": "object",
            "properties": {"task_bundle": {"type": "object"}},
            "required": ["task_bundle"],
        },
        side_effect=SideEffect.NETWORK,
        read_only=True,
        deferred=True,
        tags=["verifier", "blind-solver", "committee", "audit"],
    )

    def __init__(self, committee: VerifierCommittee):
        self.committee = committee

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        bundle = TaskBundle.from_dict(dict(arguments["task_bundle"]))
        report = self.committee.verify(bundle, context)
        return ToolResult(
            call_id,
            self.spec.name,
            "Verifier committee accepted" if report["accepted"] else "Verifier committee rejected",
            data=report,
            is_error=not report["accepted"],
            state_updates={
                "stage": "check" if report["accepted"] else "refine",
                "model_verifier_report": report,
            },
        )

