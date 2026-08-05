from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .core.store import WorkspaceLayout, atomic_write_text


def _write_skill(root: Path, manifest: dict[str, Any], body: str) -> None:
    destination = root / manifest["kind"] / manifest["skill_id"]
    destination.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination / "manifest.yaml", json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    atomic_write_text(destination / "SKILL.md", body)
    for directory in ("tools", "tests", "references", "verification"):
        (destination / directory).mkdir(exist_ok=True)
    atomic_write_text(destination / "lineage.json", json.dumps({"skill_id": manifest["skill_id"], "derivation": "demo_seed"}, indent=2))


def bootstrap_demo_workspace(workspace: str | Path) -> WorkspaceLayout:
    layout = WorkspaceLayout(Path(workspace).resolve())
    layout.initialize()
    calc_manifest = {
        "skill_id": "med.calc.two-stage-clearance", "name": "Two-stage clearance calculation",
        "version": "1.0.0", "kind": "atomic", "status": "verified",
        "description": "Construct a longitudinal case requiring the same clearance calculation at two distinct time points.",
        "intent": "Create a repeated quantitative operation with changed bindings and a temporal comparison.",
        "triggers": ["clearance", "longitudinal labs", "repeated calculation"], "negative_triggers": [],
        "inputs": ["case.baseline_value", "case.followup_value", "formula.parameters"],
        "outputs": ["case.baseline_clearance", "case.followup_clearance", "case.clearance_delta"],
        "difficulty_effect": {"base": 4, "temporal_reasoning": 2}, "tools": ["python_eval"],
        "evidence_contract": {"knowledge_dependency": False}, "composition": {},
        "source_refs": [{"type": "synthetic_demo"}], "metrics": {"passed": True},
        "tags": ["medical", "calculation", "temporal"]
    }
    calc_body = """---
name: med.calc.two-stage-clearance
---
# Skill of Two-stage Clearance Calculation

## Structure
Create one patient state at baseline and a second state after an intervention. Bind the same deterministic formula to each state using different input values.

## Action
1. State the formula and units explicitly.
2. Compute baseline and follow-up independently with `python_eval`.
3. Ask for a scalar comparison whose ground truth is uniquely determined.

## Effect
The repeated operation creates temporal reasoning without adding ungrounded medical facts.

## Failure handling
Reject zero denominators, missing units, inconsistent rounding rules, or a comparison not implied by the two calculations.
"""
    distractor_manifest = {
        "skill_id": "task.choice.numeric-distractors", "name": "Numeric distractor construction",
        "version": "1.0.0", "kind": "atomic", "status": "verified",
        "description": "Render one verified scalar answer as a single-choice task with error-derived numeric distractors.",
        "intent": "Convert deterministic calculations into robust single-choice evaluation tasks.",
        "triggers": ["single choice", "numeric answer", "distractors"], "negative_triggers": [],
        "inputs": ["gold.scalar", "calculation.intermediates"],
        "outputs": ["task.single_choice", "task.options", "task.gold_option"],
        "difficulty_effect": {"base": 2, "distractor_similarity": 2}, "tools": ["python_eval"],
        "evidence_contract": {"knowledge_dependency": False}, "composition": {},
        "source_refs": [{"type": "synthetic_demo"}], "metrics": {"passed": True},
        "tags": ["choice", "distractor", "rendering"]
    }
    distractor_body = """---
name: task.choice.numeric-distractors
---
# Skill of Numeric Distractor Construction

## Structure
Start from one externally verified scalar gold answer.

## Action
Generate distractors from distinct plausible error mechanisms: unit conversion, wrong time point, sign reversal, or premature rounding. Verify no distractor equals the gold answer after normalization.

## Effect
The task remains uniquely gradable while testing the intended calculation rather than memorized prose.

## Failure handling
Reject duplicate options, multiple equivalent answers, unexplained precision, and distractors unrelated to an identifiable error.
"""
    _write_skill(layout.skills, calc_manifest, calc_body)
    _write_skill(layout.skills, distractor_manifest, distractor_body)
    return layout

