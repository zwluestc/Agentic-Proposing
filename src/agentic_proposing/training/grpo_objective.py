from __future__ import annotations

import math
from typing import Any


def terminal_advantages(rewards: list[float], *, epsilon: float = 1e-8) -> list[float]:
    """Normalize terminal rewards within a group sampled from one prompt."""
    if len(rewards) < 2:
        raise ValueError("GRPO requires at least two rollouts per prompt")
    if epsilon <= 0 or not all(math.isfinite(value) for value in rewards):
        raise ValueError("Rewards must be finite and epsilon positive")
    mean = sum(rewards) / len(rewards)
    scale = math.sqrt(sum((value - mean) ** 2 for value in rewards) / len(rewards))
    return [(value - mean) / (scale + epsilon) for value in rewards]


def token_objective(current: Any, old: Any, reference: Any, advantage: float, *,
                    clip_epsilon: float = 0.2, beta: float = 0.05):
    """Token PPO surrogate and nonnegative sampled KL to the frozen SFT policy.

    Returns unreduced loss, KL estimate, and clipping indicator. The caller first
    averages over all action tokens in each trajectory, then over trajectories.
    Old/reference probabilities are constants even if tensors require gradients.
    """
    import torch

    if not 0 < clip_epsilon < 1 or beta < 0:
        raise ValueError("Require 0 < clip_epsilon < 1 and beta >= 0")
    if current.ndim != 1 or current.numel() == 0 or current.shape != old.shape or current.shape != reference.shape:
        raise ValueError("One matching, nonempty log-probability vector per action is required")
    if not all(bool(torch.isfinite(value).all()) for value in (current, old, reference)):
        raise ValueError("GRPO log probabilities must be finite")
    ratio = torch.exp(current - old.detach())
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
    log_reference_ratio = reference.detach() - current
    kl = torch.expm1(log_reference_ratio) - log_reference_ratio
    loss = -surrogate + beta * kl
    if not bool(torch.isfinite(loss).all()):
        raise FloatingPointError("GRPO token objective overflowed; reduce learning rate or update_epochs")
    clipped = (ratio * advantage > clipped_ratio * advantage).float()
    return loss, kl, clipped
