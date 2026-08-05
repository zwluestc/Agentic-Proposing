from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import argparse
import json
import shutil

from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.config import RuntimeConfig
from agentic_proposing.core.models import ModelTurn, ToolCall
from agentic_proposing.providers.base import TeacherProvider
from agentic_proposing.runtime import build_runtime


def _latest_tool_data(messages: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for message in reversed(messages):
        if message.get("role") != "tool" or message.get("name") != name:
            continue
        payload = json.loads(str(message.get("content", "{}")))
        return dict(payload.get("data", {}))
    raise RuntimeError(f"No result found for tool {name}")


class FullLoopCaseTeacher(TeacherProvider):
    """Deterministic stateful Teacher used to exercise the real harness and tool boundaries."""

    def __init__(self, source_path: Path):
        self.source_path = source_path
        self.step = 0
        self.evidence: dict[str, Any] = {}
        self.calculation: dict[str, Any] = {}

    def _bundle(self, *, valid: bool) -> dict[str, Any]:
        evidence = self.evidence
        return {
            "task_id": "full-loop-grounded-001",
            "task_type": "structured_output",
            "question": (
                "依据所给合成构题笔记，某候选题用 C=100/x 分别计算基线 x=4 与随访 x=5，"
                "但遗漏随访输入的单位，并在来源没有建立因果关系时声称干预导致了变化。"
                "请输出 JSON：decision 只能是 accept 或 reject_or_refine；decrease 为 C 的下降值。"
            ),
            "answer": {"decision": "reject_or_refine", "decrease": 5.0},
            "rationale": (
                "基线 C=25，随访 C=20，下降 5。来源要求每个时间点明确变量、单位与舍入，"
                "且没有证据时不得把时间点之间的事件表述为因果，因此应拒绝或细化。"
            ),
            "rubric": (
                [
                    {
                        "criterion": "decision is reject_or_refine because a unit is missing and causality is unsupported",
                        "points": 2,
                        "required": True,
                        "fatal_if_violated": True,
                    },
                    {
                        "criterion": "decrease equals 5.0 from independent baseline and follow-up calculations",
                        "points": 1,
                        "required": True,
                    },
                ]
                if valid
                else []
            ),
            "evidence": [
                {
                    "evidence_id": evidence["evidence_id"],
                    "source_id": evidence["source_id"],
                    "locator": evidence["locator"],
                    "quote": str(evidence["text"])[:900],
                    "title": evidence["title"],
                    "supports": ["unit completeness", "non-causal temporal comparison"],
                }
            ],
            "skill_lineage": [
                "med.calc.two-stage-clearance",
                "task.choice.numeric-distractors",
            ],
            "difficulty": {"estimated": 5, "components": ["evidence", "calculation", "error detection"]},
            "metadata": {
                "knowledge_dependency": True,
                "output_constraints": {"required_keys": ["decision", "decrease"]},
                "case_calculation": self.calculation,
            },
        }

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> ModelTurn:
        step = self.step
        self.step += 1
        if step == 0:
            return ModelTurn(
                content="Discover the tools needed for a grounded calculation and verification run.",
                tool_calls=[
                    ToolCall(
                        "tool_search",
                        {
                            "query": "corpus evidence skill persistent python verification",
                            "load": [
                                "corpus_ingest",
                                "evidence_search",
                                "evidence_fetch",
                                "skill_search",
                                "skill_load",
                                "python_session_create",
                                "python_session_exec",
                                "python_session_close",
                            ],
                            "limit": 20,
                        },
                    )
                ],
            )
        if step == 1:
            return ModelTurn(
                content="Ingest the supplied source into the provenance-preserving EvidenceStore.",
                tool_calls=[
                    ToolCall(
                        "corpus_ingest",
                        {
                            "path": str(self.source_path),
                            "title": "Synthetic renal reasoning notes",
                            "target_chars": 700,
                            "overlap_chars": 80,
                        },
                    )
                ],
            )
        if step == 2:
            return ModelTurn(
                content="Retrieve the exact chunk that defines repeated-state completeness and causal restraint.",
                tool_calls=[
                    ToolCall("evidence_search", {"query": "multiple time points causal", "limit": 3})
                ],
            )
        if step == 3:
            rows = _latest_tool_data(messages, "evidence_search").get("results", [])
            if not rows:
                raise RuntimeError("Evidence search unexpectedly returned no rows")
            self.evidence = dict(rows[0])
            return ModelTurn(
                content="Find reusable construction procedures instead of embedding them in the prompt.",
                tool_calls=[
                    ToolCall("skill_search", {"query": "repeated calculation numeric task", "limit": 5})
                ],
            )
        if step == 4:
            return ModelTurn(
                content="Progressively load only the two selected Skill bodies.",
                tool_calls=[
                    ToolCall(
                        "skill_load",
                        {
                            "skill_ids": [
                                "med.calc.two-stage-clearance",
                                "task.choice.numeric-distractors",
                            ]
                        },
                    )
                ],
            )
        if step == 5:
            return ModelTurn(
                content="Create a stateful analysis session for dependent calculations.",
                tool_calls=[ToolCall("python_session_create", {"session_id": "case_analysis"})],
            )
        if step == 6:
            return ModelTurn(
                content="Compute both states independently, then their decrease.",
                tool_calls=[
                    ToolCall(
                        "python_session_exec",
                        {
                            "session_id": "case_analysis",
                            "code": (
                                "baseline = 100 / 4\n"
                                "followup = 100 / 5\n"
                                "decrease = baseline - followup\n"
                                "{'baseline': baseline, 'followup': followup, 'decrease': decrease}"
                            ),
                        },
                    )
                ],
            )
        if step == 7:
            result = _latest_tool_data(messages, "python_session_exec").get("result", {})
            self.calculation = dict(result)
            invalid = self._bundle(valid=False)
            return ModelTurn(
                content="Draft the candidate, then probe it before terminal submission.",
                tool_calls=[
                    ToolCall(
                        "working_state_edit",
                        {
                            "stage": "draft",
                            "working_artifact": invalid,
                            "decision": "First draft intentionally leaves the structured-output rubric incomplete.",
                        },
                    ),
                    ToolCall("self_check_artifact", {"task_bundle": invalid}),
                ],
            )
        if step == 8:
            valid = self._bundle(valid=True)
            return ModelTurn(
                content="Use the failed rubric gate to refine the candidate and check the exact revised fingerprint.",
                tool_calls=[
                    ToolCall(
                        "working_state_edit",
                        {
                            "stage": "refine",
                            "working_artifact": valid,
                            "failed_check": {"gate": "rubric", "action": "added explicit weighted criteria"},
                            "decision": "Repair only the verifier-identified defect, retaining evidence and calculation lineage.",
                        },
                    ),
                    ToolCall("self_check_artifact", {"task_bundle": valid}),
                ],
            )
        if step == 9:
            valid = self._bundle(valid=True)
            return ModelTurn(
                content="Close analysis resources and submit the exact self-checked candidate.",
                tool_calls=[
                    ToolCall("python_session_close", {"session_id": "case_analysis"}),
                    ToolCall("submit_artifact", {"task_bundle": valid}),
                ],
            )
        raise RuntimeError("Full-loop case exceeded its expected trajectory")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a complete grounded Agentic Proposing case")
    parser.add_argument("--workspace", default="case_runs/full-loop-case")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    workspace = Path(args.workspace).resolve()
    layout = bootstrap_demo_workspace(workspace)
    source = layout.sources / "synthetic_renal_notes.md"
    shutil.copy2(project_root / "examples" / "sources" / "synthetic_renal_notes.md", source)

    runtime = build_runtime(
        workspace,
        FullLoopCaseTeacher(source),
        RuntimeConfig(
            harness={"max_turns": 14, "temperature": 0.0},
            permissions={
                "allow_read_only": True,
                "allow_network": False,
                "allow_process": True,
                "allow_registry_writes": True,
            },
            tools={"max_inline_chars": 12000},
            skill_mining={"enabled": False},
            verification={"require_self_check": True},
        ),
    )
    result = runtime.harness.run(
        "Create one evidence-grounded task that combines source-rule checking with a repeated calculation"
    )
    run_dir = runtime.layout.run_dir(result.run_id)
    events = runtime.layout.run_dir(result.run_id).joinpath("events.jsonl")
    event_rows = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines() if line]
    counts = Counter(row["event_type"] for row in event_rows)
    summary = {
        "run_id": result.run_id,
        "status": result.status,
        "turns": result.turns,
        "failed_checks": result.state.failed_checks,
        "event_counts": dict(sorted(counts.items())),
        "events": str(events),
        "final_artifact": result.terminal_result.data.get("artifact_ref") if result.terminal_result else None,
        "run_directory": str(run_dir),
    }
    summary_path = workspace / "latest-summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "summary": str(summary_path)}, ensure_ascii=False, indent=2))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
