from __future__ import annotations

import pytest

from agentic_proposing.training.data import action_text, policy_messages
from agentic_proposing.training.mgpo_objective import Curriculum, mgpo_weights, terminal_reward
from agentic_proposing.training.prober import answers_match, extract_boxed


def test_terminal_reward_only_bonuses_solvable_valid_tasks() -> None:
    assert terminal_reward(False, 0.25) == 0.0
    assert terminal_reward(True, 0.0) == 1.0
    assert terminal_reward(True, 0.25) == 1.75


def test_mgpo_group_and_stage_credit_with_zero_sum_weights() -> None:
    group = [
        {"valid": False, "terminal_reward": 0.0, "transitions": [
            {"stage": "draft", "process_reward": 0.0,
             "policy_raw": {"old_logprob": -2.0, "reference_logprob": -2.0}}
        ]},
        {"valid": True, "terminal_reward": 2.0, "transitions": [
            {"stage": "draft", "process_reward": 1.0,
             "policy_raw": {"old_logprob": -2.0, "reference_logprob": -2.0}}
        ]},
    ]
    weights = mgpo_weights(group, [-2.0, -2.0], omega=0.5)
    assert weights == pytest.approx([-1.5, 1.5])
    assert sum(weights) == pytest.approx(0.0)
    group[1]["transitions"][0]["policy_raw"]["old_logprob"] = -3.0
    gated = mgpo_weights(group, [-2.0, -2.0], omega=0.5)
    assert 0.0 < gated[1] < weights[1]


def test_curriculum_shifts_toward_lower_mastery() -> None:
    curriculum = Curriculum(["A", "B"], alpha=0.5, initial=0.5)
    curriculum.update("A", [True, True])
    curriculum.update("B", [False, False])
    probabilities = curriculum.probabilities()
    assert probabilities["B"] > probabilities["A"]


def test_observation_serializes_native_tool_history() -> None:
    observation = [
        {"role": "system", "content": "Propose a task."},
        {"role": "assistant", "content": "Check units.", "tool_calls": [
            {"function": {"name": "python_eval", "arguments": '{"expression":"2+2"}'}}
        ]},
        {"role": "tool", "name": "python_eval", "content": "4"},
    ]
    messages = policy_messages(observation, [{"name": "python_eval"}])
    assert "Policy action protocol" in messages[0]["content"]
    assert "python_eval" in messages[1]["content"]
    assert messages[2] == {"role": "user", "content": "Tool result (python_eval): 4"}
    assert action_text({"content": "done", "tool_calls": []}) == '{"content":"done","tool_calls":[]}'


def test_boxed_fraction_rule_grader() -> None:
    assert extract_boxed("Work... \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert answers_match("\\boxed{\\frac{1}{2}}", "0.5")


def test_closed_loop_observer_captures_executed_actions(tmp_path) -> None:
    from agentic_proposing.bootstrap import bootstrap_demo_workspace
    from agentic_proposing.config import RuntimeConfig
    from agentic_proposing.core.models import ModelTurn, ToolCall
    from agentic_proposing.providers.scripted import ScriptedTeacher
    from agentic_proposing.runtime import build_runtime

    bootstrap_demo_workspace(tmp_path)
    bundle = {
        "task_id": "observer-demo",
        "task_type": "single_choice",
        "question": "A synthetic formula gives 25 at baseline and 20 at follow-up. What is the decrease?",
        "answer": "5",
        "metadata": {"knowledge_dependency": False, "options": ["3", "4", "5", "6"]},
        "skill_lineage": ["med.calc.two-stage-clearance", "task.choice.numeric-distractors"],
    }
    teacher = ScriptedTeacher([
        ModelTurn(tool_calls=[ToolCall("self_check_artifact", {"task_bundle": bundle})]),
        ModelTurn(tool_calls=[ToolCall("submit_artifact", {"task_bundle": bundle})]),
    ])
    observed = []
    runtime = build_runtime(
        tmp_path,
        teacher,
        RuntimeConfig(verification={"require_self_check": True}, skill_mining={"enabled": False}),
        action_observer=observed.append,
    )
    result = runtime.harness.run("Create a calculation task")
    assert result.status == "completed"
    assert [row["action"]["tool_calls"][0]["name"] for row in observed] == [
        "self_check_artifact", "submit_artifact"
    ]
    assert observed[-1]["tool_results"][0]["terminal"] is True
