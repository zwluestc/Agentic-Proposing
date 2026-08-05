from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol
import json
import re

from agentic_proposing.core.models import TaskBundle, jsonable
from agentic_proposing.skills.dedup import jaccard


@dataclass(slots=True)
class CheckResult:
    name: str
    passed: bool
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GateReport:
    passed: bool
    checks: list[CheckResult]

    def to_dict(self) -> dict[str, Any]:
        return jsonable(self)


class VerificationContext(Protocol):
    services: dict[str, Any]


class Gate(Protocol):
    name: str
    def check(self, bundle: TaskBundle, context: VerificationContext) -> CheckResult: ...


class SchemaGate:
    name = "schema"

    def check(self, bundle: TaskBundle, context: VerificationContext) -> CheckResult:
        errors: list[str] = []
        if not bundle.question.strip():
            errors.append("question is empty")
        if bundle.answer is None or bundle.answer == "":
            errors.append("answer is empty")
        if not bundle.task_type.strip():
            errors.append("task_type is empty")
        if len(bundle.question) < 12:
            errors.append("question is too short to be well-specified")
        if bundle.task_type in {"single_choice", "multiple_choice"}:
            options = bundle.metadata.get("options", [])
            if not isinstance(options, list) or len(options) < 2:
                errors.append("choice task requires at least two options in metadata.options")
        return CheckResult(self.name, not errors, "fatal", "; ".join(errors) if errors else "schema valid")


class RubricGate:
    name = "rubric"

    def check(self, bundle: TaskBundle, context: VerificationContext) -> CheckResult:
        errors: list[str] = []
        if bundle.task_type in {"open_qa", "summary", "structured_output", "rubric_scored"} and not bundle.rubric:
            errors.append("open/rubric task requires rubric items")
        if bundle.rubric:
            total = sum(item.points for item in bundle.rubric)
            if total <= 0:
                errors.append("rubric total points must be positive")
            criteria = [item.criterion.strip().lower() for item in bundle.rubric]
            if len(criteria) != len(set(criteria)):
                errors.append("rubric contains duplicate criteria")
        return CheckResult(self.name, not errors, "fatal", "; ".join(errors) if errors else "rubric valid")


class EvidenceGate:
    name = "evidence"

    def check(self, bundle: TaskBundle, context: VerificationContext) -> CheckResult:
        knowledge_dependency = bool(bundle.metadata.get("knowledge_dependency", True))
        if not knowledge_dependency:
            return CheckResult(self.name, True, "fatal", "task declares no external knowledge dependency")
        if not bundle.evidence:
            return CheckResult(self.name, False, "fatal", "knowledge-dependent task has no evidence")
        missing = []
        store = context.services.get("evidence_store")
        for ref in bundle.evidence:
            if not ref.evidence_id or not ref.source_id or not ref.locator:
                missing.append(ref.evidence_id or "<missing-id>")
                continue
            if store:
                try:
                    store.fetch(ref.evidence_id)
                except KeyError:
                    missing.append(ref.evidence_id)
        if missing:
            return CheckResult(
                self.name, False, "fatal", "evidence references are unresolved", {"missing": missing}
            )
        return CheckResult(self.name, True, "fatal", "evidence references resolved")


class GroundTruthGate:
    name = "ground_truth"

    def check(self, bundle: TaskBundle, context: VerificationContext) -> CheckResult:
        if bundle.task_type == "single_choice":
            options = bundle.metadata.get("options", [])
            answer = bundle.answer
            valid = answer in options or (
                isinstance(answer, str) and len(answer) == 1 and answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(options)]
            )
            return CheckResult(
                self.name,
                valid,
                "fatal",
                "single-choice answer resolves to one option" if valid else "single-choice answer does not resolve to one option",
            )
        if bundle.task_type in {"fill_blank", "calculation"} and isinstance(bundle.answer, (list, dict)):
            return CheckResult(self.name, False, "fatal", "unique-ground-truth task has non-scalar answer")
        return CheckResult(self.name, True, "fatal", "ground-truth shape is compatible with task type")


class SkillLineageGate:
    name = "skill_lineage"

    def check(self, bundle: TaskBundle, context: VerificationContext) -> CheckResult:
        registry = context.services.get("skill_registry")
        staged_ids = set(context.services.get("staged_skill_ids", []))
        layout = getattr(context, "layout", None)
        if layout is not None and layout.staging.joinpath("skills").exists():
            staged_ids.update(
                path.name.split("@", 1)[0]
                for path in layout.staging.joinpath("skills").iterdir()
                if path.is_dir()
            )
        missing = [
            skill_id for skill_id in bundle.skill_lineage
            if registry and not registry.has(skill_id) and skill_id not in staged_ids
        ]
        if missing:
            return CheckResult(self.name, False, "fatal", "unknown skills in lineage", {"missing": missing})
        return CheckResult(self.name, True, "fatal", "skill lineage resolved")


class ContaminationGate:
    name = "contamination"

    def __init__(self, threshold: float = 0.88):
        self.threshold = threshold

    def check(self, bundle: TaskBundle, context: VerificationContext) -> CheckResult:
        references = context.services.get("contamination_questions", [])
        best = {"score": 0.0, "reference": ""}
        for reference in references:
            score = jaccard(bundle.question, str(reference))
            if score > best["score"]:
                best = {"score": score, "reference": str(reference)[:500]}
        passed = best["score"] < self.threshold
        return CheckResult(
            self.name,
            passed,
            "fatal",
            "no near-copy detected" if passed else "question is too similar to a protected source item",
            best,
        )


class OutputConstraintGate:
    name = "output_constraints"

    def check(self, bundle: TaskBundle, context: VerificationContext) -> CheckResult:
        constraints = bundle.metadata.get("output_constraints", {})
        errors: list[str] = []
        answer_text = bundle.answer if isinstance(bundle.answer, str) else json.dumps(bundle.answer, ensure_ascii=False)
        if "max_chars" in constraints and len(answer_text) > int(constraints["max_chars"]):
            errors.append("answer exceeds max_chars")
        if "required_keys" in constraints:
            if not isinstance(bundle.answer, dict):
                errors.append("required_keys specified but answer is not an object")
            else:
                absent = [key for key in constraints["required_keys"] if key not in bundle.answer]
                if absent:
                    errors.append(f"answer missing required keys: {absent}")
        return CheckResult(self.name, not errors, "fatal", "; ".join(errors) if errors else "output constraints satisfied")


class VerificationPipeline:
    def __init__(self, gates: list[Gate] | None = None):
        self.gates: list[Gate] = gates or [
            SchemaGate(),
            RubricGate(),
            EvidenceGate(),
            GroundTruthGate(),
            SkillLineageGate(),
            OutputConstraintGate(),
            ContaminationGate(),
        ]

    def verify(self, bundle: TaskBundle, context: VerificationContext) -> GateReport:
        checks: list[CheckResult] = []
        for gate in self.gates:
            try:
                result = gate.check(bundle, context)
            except Exception as exc:
                result = CheckResult(gate.name, False, "fatal", f"gate error: {type(exc).__name__}: {exc}")
            checks.append(result)
        passed = all(check.passed or check.severity != "fatal" for check in checks)
        return GateReport(passed, checks)

