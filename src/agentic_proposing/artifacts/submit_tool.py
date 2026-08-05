from __future__ import annotations

from typing import Any

from agentic_proposing.core.hooks import HookEvent
from agentic_proposing.core.models import Event, SideEffect, TaskBundle, ToolResult
from agentic_proposing.tools.base import BaseTool, ToolContext, ToolSpec
from .self_check_tool import bundle_fingerprint
from .verification import VerificationPipeline


class SubmitArtifactTool(BaseTool):
    spec = ToolSpec(
        name="submit_artifact",
        description=(
            "Submit a complete TaskBundle through deterministic and configured verifier gates. Rejection returns structured "
            "failures and the agent must refine; acceptance is the only successful terminal action."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task_bundle": {
                    "type": "object",
                    "properties": {
                        "task_id": {"type": "string"},
                        "task_type": {"type": "string"},
                        "question": {"type": "string"},
                        "answer": {},
                        "rationale": {"type": "string"},
                        "rubric": {"type": "array", "items": {"type": "object"}},
                        "evidence": {"type": "array", "items": {"type": "object"}},
                        "skill_lineage": {"type": "array", "items": {"type": "string"}},
                        "difficulty": {"type": "object"},
                        "metadata": {"type": "object"},
                    },
                    "required": ["task_id", "task_type", "question", "answer"],
                }
            },
            "required": ["task_bundle"],
        },
        side_effect=SideEffect.WORKSPACE,
        read_only=False,
        deferred=False,
        tags=["submit", "finalize", "verify", "artifact"],
    )

    def __init__(self, pipeline: VerificationPipeline | None = None, *, require_self_check: bool = False):
        self.pipeline = pipeline or VerificationPipeline()
        self.require_self_check = require_self_check

    def invoke(self, arguments: dict[str, Any], context: ToolContext, call_id: str) -> ToolResult:
        bundle = TaskBundle.from_dict(dict(arguments["task_bundle"]))
        if self.require_self_check:
            current = bundle_fingerprint(bundle)
            checked = str(context.state.metadata.get("self_check_fingerprint", ""))
            if not checked or checked != current:
                reason = "A passing self_check_artifact is required" if not checked else "Candidate changed after self-check"
                return ToolResult(
                    call_id,
                    self.spec.name,
                    reason + "; run self_check_artifact on the exact candidate before submitting",
                    data={"expected_fingerprint": current, "checked_fingerprint": checked or None},
                    is_error=True,
                    state_updates={
                        "stage": "refine",
                        "working_artifact": bundle.to_dict(),
                        "append_failed_check": {
                            "gate": "self_check_freshness",
                            "message": reason,
                            "expected_fingerprint": current,
                            "checked_fingerprint": checked or None,
                        },
                    },
                )
        hooks = context.services.get("hooks")
        if hooks:
            decision = hooks.emit(HookEvent.PRE_SUBMIT, {"bundle": bundle.to_dict(), "run_id": context.state.run_id})
            if not decision.allow:
                return ToolResult(
                    call_id,
                    self.spec.name,
                    decision.reason or "Submission blocked by hook",
                    is_error=True,
                    state_updates={
                        "stage": "refine",
                        "append_failed_check": {"gate": "pre_submit_hook", "message": decision.reason},
                    },
                )
        report = self.pipeline.verify(bundle, context)
        bundle.verification["gates"] = report.to_dict()
        candidate_ref = context.run_store.write_artifact(
            f"candidate-{bundle.task_id}.json", bundle.to_dict(), "artifacts"
        )
        if not report.passed:
            failures = [check for check in report.to_dict()["checks"] if not check["passed"]]
            return ToolResult(
                call_id,
                self.spec.name,
                "TaskBundle rejected; refine using failed checks",
                data={"report": report.to_dict(), "candidate_ref": candidate_ref},
                is_error=True,
                state_updates={
                    "stage": "refine",
                    "working_artifact": bundle.to_dict(),
                    "append_failed_check": {"gate": "submit", "failures": failures},
                },
                artifact_refs=[candidate_ref],
            )
        final_ref = context.run_store.write_artifact(
            f"accepted-{bundle.task_id}.json", bundle.to_dict(), "final"
        )
        mining_result: dict[str, Any] = {}
        post_accept = context.services.get("post_accept_pipeline")
        if post_accept:
            try:
                mining_result = post_accept.run(bundle, context)
            except Exception as exc:
                mining_result = {
                    "staged": [],
                    "rejected": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                context.run_store.events.append(
                    Event(
                        "post_accept_skill_mining_failed",
                        context.state.run_id,
                        mining_result,
                    )
                )
        if hooks:
            hooks.emit(
                HookEvent.POST_SUBMIT,
                {"bundle": bundle.to_dict(), "run_id": context.state.run_id, "artifact_ref": final_ref},
            )
        return ToolResult(
            call_id,
            self.spec.name,
            "TaskBundle accepted",
            data={"report": report.to_dict(), "artifact_ref": final_ref, "skill_mining": mining_result},
            terminal=True,
            state_updates={
                "stage": "complete",
                "working_artifact": bundle.to_dict(),
                "accepted_artifact": final_ref,
            },
            artifact_refs=[candidate_ref, final_ref],
        )

