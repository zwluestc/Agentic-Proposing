from __future__ import annotations

from typing import Any
import hashlib
import json

from agentic_proposing.core.models import SideEffect, TaskBundle, ToolResult
from agentic_proposing.tools.base import BaseTool, ToolContext, ToolSpec
from .verification import VerificationPipeline


def bundle_fingerprint(bundle: TaskBundle) -> str:
    payload = bundle.to_dict()
    payload.pop("verification", None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SelfCheckArtifactTool(BaseTool):
    spec = ToolSpec(
        name="self_check_artifact",
        description=(
            "Run deterministic verifier gates without ending the agent loop. A passing check records a fingerprint; "
            "a failing check returns structured defects for refinement. Run again after every material edit."
        ),
        input_schema={
            "type": "object",
            "properties": {"task_bundle": {"type": "object"}},
            "required": ["task_bundle"],
        },
        side_effect=SideEffect.WORKSPACE,
        read_only=False,
        deferred=False,
        tags=["self-check", "verify", "artifact", "refine", "nonterminal"],
    )

    def __init__(self, pipeline: VerificationPipeline | None = None):
        self.pipeline = pipeline or VerificationPipeline()

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        bundle = TaskBundle.from_dict(dict(arguments["task_bundle"]))
        report = self.pipeline.verify(bundle, context)
        fingerprint = bundle_fingerprint(bundle)
        payload = {
            "task_id": bundle.task_id,
            "fingerprint": fingerprint,
            "report": report.to_dict(),
        }
        ref = context.run_store.write_artifact(
            f"self-check-{bundle.task_id}-{fingerprint[:10]}.json", payload, "artifacts"
        )
        failures = [item for item in report.to_dict()["checks"] if not item["passed"]]
        updates: dict[str, Any] = {
            "stage": "check" if report.passed else "refine",
            "working_artifact": bundle.to_dict(),
            "self_check_fingerprint": fingerprint if report.passed else "",
            "self_check_report": report.to_dict(),
            "append_artifact_ref": ref,
        }
        if failures:
            updates["append_failed_check"] = {"gate": "self_check", "failures": failures}
        return ToolResult(
            call_id,
            self.spec.name,
            "Self-check passed; candidate may be submitted" if report.passed else "Self-check failed; refine candidate",
            data={**payload, "artifact_ref": ref, "failures": failures},
            is_error=not report.passed,
            state_updates=updates,
            artifact_refs=[ref],
        )
