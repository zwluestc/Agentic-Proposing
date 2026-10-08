from __future__ import annotations

from collections import defaultdict
from typing import Any
import math


def terminal_reward(valid: bool, rho: float, base: float = 1.0, difficulty_weight: float = 1.0) -> float:
    """Paper Eq. 7. A zero prober pass rate earns no difficulty bonus."""
    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    return float(valid) * (base + (difficulty_weight * (1.0 - rho) if rho > 0.0 else 0.0))


def _standardize(values: list[float]) -> list[float]:
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    if variance < 1e-12:
        return [0.0] * len(values)
    scale = math.sqrt(variance)
    return [(x - mean) / scale for x in values]


def _sech_squared(value: float) -> float:
    if abs(value) > 30.0:
        return 0.0
    return 1.0 / math.cosh(value) ** 2


def mgpo_weights(trajectories: list[dict[str, Any]], current_logprobs: list[float], *,
                 omega: float = 0.5, beta: float = 0.05,
                 tau_pos: float = 1.0, tau_neg: float = 1.05) -> list[float]:
    """Paper Eqs. 8-10: group and stage normalization, implicit KL, asymmetric gate.

    `trajectories` is one rollout group for a single synthesis objective. The returned
    per-action weights are treated as constants during the weighted MLE step.
    """
    if tau_neg <= tau_pos or tau_pos <= 0:
        raise ValueError("The MGPO gate requires tau_neg > tau_pos > 0")
    if omega < 0 or beta <= 0:
        raise ValueError("omega must be nonnegative and beta positive")
    actions: list[tuple[int, dict[str, Any]]] = []
    for i, trajectory in enumerate(trajectories):
        actions.extend((i, transition) for transition in trajectory["transitions"])
    if len(actions) != len(current_logprobs):
        raise ValueError("One current action log probability is required per transition")
    terminal = [float(t["terminal_reward"]) for t in trajectories]
    trajectory_advantage = _standardize(terminal)
    stage_members: dict[str, list[int]] = defaultdict(list)
    for index, (_, action) in enumerate(actions):
        stage_members[str(action["stage"])].append(index)
    process_advantage = [0.0] * len(actions)
    for indexes in stage_members.values():
        rewards = [
            float(actions[index][1].get("process_reward", 0.0))
            if bool(trajectories[actions[index][0]].get("valid", False)) else 0.0
            for index in indexes
        ]
        for index, value in zip(indexes, _standardize(rewards)):
            process_advantage[index] = value
    fused = [trajectory_advantage[i] + omega * process_advantage[index]
             for index, (i, _) in enumerate(actions)]
    implicit = [
        beta * (current_logprobs[index] - float(action["policy_raw"]["reference_logprob"]))
        for index, (_, action) in enumerate(actions)
    ]
    fused_centered = [0.0] * len(actions)
    implicit_centered = [0.0] * len(actions)
    for indexes in stage_members.values():
        mean_fused = sum(fused[i] for i in indexes) / len(indexes)
        mean_implicit = sum(implicit[i] for i in indexes) / len(indexes)
        for index in indexes:
            fused_centered[index] = fused[index] - mean_fused
            implicit_centered[index] = implicit[index] - mean_implicit
    weights: list[float] = []
    for index, (_, action) in enumerate(actions):
        weight = fused_centered[index] - implicit_centered[index]
        tau = tau_pos if weight >= 0.0 else tau_neg
        log_ratio = current_logprobs[index] - float(action["policy_raw"]["old_logprob"])
        ratio = math.exp(max(-20.0, min(20.0, log_ratio)))
        gate = _sech_squared(tau * (ratio - 1.0) / 2.0)
        weights.append(gate * weight)
    return weights


class Curriculum:
    """Paper Eq. 6: EMA mastery and inverse-proficiency category sampling."""

    def __init__(self, categories: list[str], *, alpha: float = 0.1, epsilon: float = 0.05,
                 initial: float = 0.5):
        if not categories or not 0 < alpha < 1 or epsilon <= 0:
            raise ValueError("Invalid curriculum configuration")
        self.alpha = alpha
        self.epsilon = epsilon
        self.mastery = {category: initial for category in sorted(set(categories))}

    def probabilities(self) -> dict[str, float]:
        raw = {category: 1.0 / (mastery + self.epsilon)
               for category, mastery in self.mastery.items()}
        total = sum(raw.values())
        return {category: value / total for category, value in raw.items()}

    def update(self, category: str, successes: list[bool]) -> None:
        if not successes:
            return
        previous = self.mastery[category]
        self.mastery[category] = (1.0 - self.alpha) * previous + self.alpha * (
            sum(bool(value) for value in successes) / len(successes)
        )
