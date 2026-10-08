from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json
import os
import random
import re
import shutil

from agentic_proposing.config import RuntimeConfig, create_provider
from agentic_proposing.runtime import build_runtime
from .data import read_jsonl, seed_objective, seed_run_state, write_jsonl
from .local_policy import LocalPolicyProvider, action_logprob, action_token_logprobs, load_model
from .grpo_objective import terminal_advantages, token_objective
from .mgpo_objective import Curriculum, mgpo_weights, terminal_reward
from .prober import FrozenProber


def _process_reward(transition: dict[str, Any], verifier: Any, *, judge_reflections: bool) -> float:
    tool_success = any(
        not result.get("is_error", True)
        and result.get("tool_name") not in {"submit_artifact", "verifier_committee"}
        for result in transition["tool_results"]
    )
    reward = float(tool_success)
    content = str(transition["action"].get("content", "")).strip()
    if judge_reflections and content:
        verdict = verifier.complete_json([{
            "role": "user",
            "content": (
                "Judge whether this concise agent reflection is logically coherent for its "
                "synthesis stage. Return strict JSON {\"coherent\":true|false}. "
                "Do not reveal or request hidden reasoning.\n"
                + json.dumps({"stage": transition["stage"], "reflection": content}, ensure_ascii=False)
            ),
        }], temperature=0.0, max_tokens=128)
        if isinstance(verdict, dict) and verdict.get("coherent") is True:
            reward += 1.0
    return reward


def _flat_actions(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [action for trajectory in group for action in trajectory["transitions"]]


def _update_policy(policy: Any, optimizer: Any, group: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, float]:
    import torch

    actions = _flat_actions(group)
    if not actions:
        return {"loss": 0.0, "actions": 0.0, "nonzero_weights": 0.0}
    policy.eval()
    with torch.no_grad():
        current = [
            float(action_logprob(
                policy, action["policy_raw"]["prompt_token_ids"],
                action["policy_raw"]["completion_token_ids"],
            ).item())
            for action in actions
        ]
    weights = mgpo_weights(
        group, current,
        omega=float(settings.get("omega", 0.5)),
        beta=float(settings.get("beta", 0.05)),
        tau_pos=float(settings.get("tau_pos", 1.0)),
        tau_neg=float(settings.get("tau_neg", 1.05)),
    )
    total_tokens = sum(len(action["policy_raw"]["completion_token_ids"]) for action in actions)
    optimizer.zero_grad(set_to_none=True)
    objective = 0.0
    policy.train()
    policy.config.use_cache = False
    for action, weight in zip(actions, weights):
        if abs(weight) < 1e-10:
            continue
        logprob = action_logprob(
            policy, action["policy_raw"]["prompt_token_ids"],
            action["policy_raw"]["completion_token_ids"],
        )
        loss = -(weight / total_tokens) * logprob
        loss.backward()
        objective += float(loss.detach().item())
    nonzero = sum(abs(weight) >= 1e-10 for weight in weights)
    if nonzero:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings.get("max_grad_norm", 1.0)))
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": objective, "actions": float(len(actions)), "nonzero_weights": float(nonzero)}


def _update_grpo_policy(policy: Any, optimizer: Any, group: list[dict[str, Any]],
                        settings: dict[str, Any]) -> dict[str, float]:
    """Apply terminal-only GRPO over sampled policy tokens across all turns."""
    import torch

    advantages = terminal_advantages([float(row["terminal_reward"]) for row in group])
    epochs = int(settings.get("update_epochs", 1))
    if epochs < 1:
        raise ValueError("GRPO update_epochs must be positive")
    # Preserve eval-time probabilities while retaining training-mode checkpointing.
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    policy.train()
    policy.config.use_cache = False
    metrics = {"loss": 0.0, "kl": 0.0, "clip_fraction": 0.0}
    actions = _flat_actions(group)
    if not actions:
        return {**metrics, "actions": 0.0, "tokens": 0.0, "advantage_std": 0.0}
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for trajectory, advantage in zip(group, advantages):
            transitions = trajectory["transitions"]
            token_count = sum(len(action["policy_raw"]["completion_token_ids"]) for action in transitions)
            if not token_count:
                continue
            normalizer = token_count * len(group)
            for action in transitions:
                sample = action["policy_raw"]
                current = action_token_logprobs(
                    policy, sample["prompt_token_ids"], sample["completion_token_ids"],
                    temperature=float(sample["logprob_temperature"]),
                )
                old = torch.as_tensor(sample["old_token_logprobs"], device=current.device, dtype=current.dtype)
                reference = torch.as_tensor(sample["reference_token_logprobs"], device=current.device, dtype=current.dtype)
                losses, kl, clipped = token_objective(
                    current, old, reference, advantage,
                    clip_epsilon=float(settings.get("clip_epsilon", 0.2)),
                    beta=float(settings.get("beta", 0.05)),
                )
                loss = losses.sum() / normalizer
                loss.backward()
                metrics["loss"] += float(loss.detach().item()) / epochs
                metrics["kl"] += float(kl.detach().sum().item()) / normalizer / epochs
                metrics["clip_fraction"] += float(clipped.sum().item()) / normalizer / epochs
        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(settings.get("max_grad_norm", 1.0)))
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        **metrics, "actions": float(len(actions)),
        "tokens": float(sum(len(action["policy_raw"]["completion_token_ids"]) for action in actions)),
        "advantage_std": (sum(value ** 2 for value in advantages) / len(advantages)) ** 0.5,
    }


def _resume_settings(settings: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in settings.items()
            if key not in {"resume_from", "iterations", "output_dir", "save_steps", "save_total_limit"}}


def _validate_resume(saved: dict[str, Any], settings: dict[str, Any], algorithm: str,
                     sft_checkpoint: Path) -> None:
    # Legacy MGPO checkpoints predate algorithm/settings metadata.
    if saved.get("algorithm", "mgpo") != algorithm:
        raise ValueError(f"Cannot resume a {saved.get('algorithm', 'mgpo')} checkpoint as {algorithm}")
    if Path(saved["reference_checkpoint"]).resolve() != sft_checkpoint.resolve():
        raise ValueError("Resume requires the original SFT reference checkpoint")
    if "settings" in saved and saved["settings"] != _resume_settings(settings):
        raise ValueError("Resume settings differ from checkpoint; only iterations/output_dir/checkpoint cadence may change")


def _prune_checkpoints(output: Path, algorithm: str, limit: int, *, protected: list[Path]) -> None:
    """Retain recent checkpoints without deleting unrelated or referenced data."""
    if limit < 1:
        raise ValueError("save_total_limit must be positive")
    protected_paths = [path.resolve() for path in protected]
    candidates: list[tuple[int, Path]] = []
    for path in output.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if not match or path.is_symlink() or not path.is_dir():
            continue
        state_path = path / "trainer_state.json"
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        iteration = int(match.group(1))
        if isinstance(state, dict) and state.get("algorithm") == algorithm and state.get("iteration") == iteration:
            candidates.append((iteration, path))
    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, path in candidates[limit:]:
        resolved = path.resolve()
        if any(resolved == keep or resolved in keep.parents for keep in protected_paths):
            continue
        shutil.rmtree(path)


def _restore_rollout_log(path: Path, iteration: int, algorithm: str) -> None:
    """Archive updates newer than the resumed weights so active logs stay aligned."""
    if not path.exists() or path.stat().st_size == 0:
        return
    rows = read_jsonl(path)
    if any(row.get("algorithm", "mgpo") != algorithm for row in rows):
        raise ValueError("The output rollout log belongs to a different algorithm")
    superseded = [row for row in rows if int(row["iteration"]) > iteration]
    if not superseded:
        return
    retained = [row for row in rows if int(row["iteration"]) <= iteration]
    archive = path.with_name("rollouts.superseded.jsonl")
    write_jsonl(archive, superseded, append=True)
    temporary = path.with_suffix(".jsonl.resume.tmp")
    write_jsonl(temporary, retained)
    temporary.replace(path)
    print(json.dumps({"resumed_iteration": iteration, "superseded_rollouts": len(superseded),
                      "superseded_archive": str(archive)}), flush=True)


def _check_fresh_output(output: Path) -> None:
    rollout_path = output / "rollouts.jsonl"
    has_rollouts = rollout_path.exists() and rollout_path.stat().st_size > 0
    has_checkpoint = any(re.fullmatch(r"checkpoint-\d+", path.name)
                         for path in output.glob("checkpoint-*"))
    if has_rollouts or has_checkpoint:
        raise ValueError(
            f"Output directory already contains rollout/checkpoint data: {output}; "
            "use --resume-from or a new --output-dir"
        )


def train_mgpo(raw: dict[str, Any], runtime_config: RuntimeConfig) -> Path:
    return _train_rl(raw, runtime_config, algorithm="mgpo")


def train_grpo(raw: dict[str, Any], runtime_config: RuntimeConfig) -> Path:
    return _train_rl(raw, runtime_config, algorithm="grpo")


def _train_rl(raw: dict[str, Any], runtime_config: RuntimeConfig, *, algorithm: str) -> Path:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Proposer GRPO/MGPO currently support one process on one CUDA device; do not use torchrun")
    settings = dict(raw[algorithm])
    save_steps = int(settings.get("save_steps", 10))
    save_total_limit = int(settings.get("save_total_limit", 2))
    if save_steps < 1 or save_total_limit < 1:
        raise ValueError("save_steps and save_total_limit must be positive")
    if int(settings.get("group_size", 8)) < 2 or int(settings.get("iterations", 10)) < 1:
        raise ValueError("RL requires group_size >= 2 and iterations >= 1")
    if algorithm == "grpo":
        if float(settings.get("rollout_temperature", 0.7)) <= 0:
            raise ValueError("GRPO rollout_temperature must be positive")
        if not 0 < float(settings.get("clip_epsilon", 0.2)) < 1 or float(settings.get("beta", 0.05)) < 0:
            raise ValueError("GRPO requires 0 < clip_epsilon < 1 and beta >= 0")
        if int(settings.get("update_epochs", 1)) < 1:
            raise ValueError("GRPO update_epochs must be positive")
    output = Path(settings["output_dir"]).resolve()
    resume = settings.get("resume_from")
    if not resume:
        _check_fresh_output(output)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Install training dependencies: python3 -m pip install -e ".[training]"') from exc
    from .runner import _prepare_workspace

    workspace = _prepare_workspace(raw)
    sft_checkpoint = Path(raw["sft"].get("checkpoint", str(Path(raw["sft"]["output_dir"]) / "final"))).resolve()
    if not sft_checkpoint.exists():
        raise FileNotFoundError(f"SFT checkpoint missing: {sft_checkpoint}")
    output.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Any] = {}
    if resume:
        saved = json.loads((Path(resume) / "trainer_state.json").read_text(encoding="utf-8"))
        _validate_resume(saved, settings, algorithm, sft_checkpoint)
        if saved.get("seed_sha256") not in {None, hashlib.sha256(Path(raw["seeds"]).read_bytes()).hexdigest()}:
            raise ValueError("Resume seed prompt contents differ from checkpoint")
        if saved.get("verifier_model") not in {None, raw["verifier"].get("model")}:
            raise ValueError("Resume verifier model differs from checkpoint")
    policy_source = str(Path(resume).resolve()) if resume else str(sft_checkpoint)
    policy, tokenizer = load_model(policy_source, trainable=True)
    reference, _ = load_model(str(sft_checkpoint), trainable=False)
    prober_model, prober_tokenizer = load_model(
        str(settings.get("prober_model", raw["sft"]["base_model"])), trainable=False
    )
    prober = FrozenProber(
        prober_model, prober_tokenizer,
        attempts=int(settings.get("prober_attempts", 16)),
        temperature=float(settings.get("prober_temperature", 1.4)),
        max_new_tokens=int(settings.get("prober_max_new_tokens", 256)),
    )
    actor = LocalPolicyProvider(
        policy, tokenizer, reference=reference,
        max_context_tokens=int(settings.get("max_context_tokens", 8192)),
        max_action_tokens=int(settings.get("max_action_tokens", 768)),
        record_token_logprobs=algorithm == "grpo",
    )
    verifier = create_provider(dict(raw["verifier"]))
    current: list[dict[str, Any]] = []

    def observe(transition: dict[str, Any]) -> None:
        current.append(transition)

    runtime_config.harness["temperature"] = float(settings.get("rollout_temperature", 0.7))
    runtime_config.harness["max_tokens_per_turn"] = int(settings.get("max_action_tokens", 768))
    runtime = build_runtime(
        workspace, actor, runtime_config,
        verifier_provider=verifier, action_observer=observe,
    )
    seeds = read_jsonl(raw["seeds"])
    if not seeds:
        raise ValueError("RL requires at least one seed prompt")
    categories = [str(seed.get("category", "general")) for seed in seeds]
    curriculum = Curriculum(
        categories, alpha=float(settings.get("curriculum_alpha", 0.1)),
        epsilon=float(settings.get("curriculum_epsilon", 0.05)),
    )
    start = 1
    if resume:
        curriculum.mastery.update(saved["mastery"])
        start = int(saved["iteration"]) + 1
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=float(settings.get("learning_rate", 1e-6)),
        betas=(0.9, 0.95), weight_decay=float(settings.get("weight_decay", 0.1)),
    )
    random.seed(int(settings.get("seed", 42)))
    torch.manual_seed(int(settings.get("seed", 42)))
    if resume:
        saved_optimizer = torch.load(
            Path(resume) / "optimizer.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(saved_optimizer["optimizer"])
        random.setstate(saved_optimizer["random_state"])
        if "torch_rng_state" in saved_optimizer:
            torch.set_rng_state(saved_optimizer["torch_rng_state"])
        if "cuda_rng_state" in saved_optimizer:
            torch.cuda.set_rng_state_all(saved_optimizer["cuda_rng_state"])
    rollout_path = output / "rollouts.jsonl"
    if resume:
        _restore_rollout_log(rollout_path, int(saved["iteration"]), algorithm)
    checkpoint = Path(resume).resolve() if resume else output
    for iteration in range(start, int(settings.get("iterations", 10)) + 1):
        probabilities = curriculum.probabilities()
        category = random.choices(list(probabilities), weights=list(probabilities.values()), k=1)[0]
        eligible = [seed for seed in seeds if str(seed.get("category", "general")) == category]
        seed = random.choice(eligible)
        objective = seed_objective(seed)
        group: list[dict[str, Any]] = []
        for member in range(int(settings.get("group_size", 8))):
            current.clear()
            result = runtime.harness.run(
                objective, seed_run_state(seed, objective, workspace, runtime.registry)
            )
            valid = result.status == "completed"
            bundle = dict(result.state.working_artifact) if valid else {}
            rho = prober.pass_rate(bundle) if valid else 0.0
            group.append({
                "algorithm": algorithm,
                "iteration": iteration,
                "member": member,
                "run_id": result.run_id,
                "seed_id": str(seed.get("id", "")),
                "category": category,
                "valid": valid,
                "rho": rho,
                "terminal_reward": terminal_reward(
                    valid, rho,
                    base=float(settings.get("reward_base", 1.0)),
                    difficulty_weight=float(settings.get("difficulty_weight", 1.0)),
                ),
                "task_bundle": bundle,
                "transitions": [dict(action, process_reward=(
                    _process_reward(action, verifier, judge_reflections=bool(settings.get("judge_reflections", True)))
                    if algorithm == "mgpo" else 0.0
                )) for action in current],
            })
            print(f"{algorithm.upper()} {iteration} rollout {member + 1}: valid={valid} rho={rho:.3f}", flush=True)
        curriculum.update(category, [bool(item["valid"]) for item in group])
        updater = _update_grpo_policy if algorithm == "grpo" else _update_policy
        metrics = updater(policy, optimizer, group, settings)
        write_jsonl(rollout_path, group, append=True)
        saved_this_iteration = iteration % save_steps == 0 or iteration == int(settings.get("iterations", 10))
        if saved_this_iteration:
            checkpoint = output / f"checkpoint-{iteration:04d}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            torch.save({"optimizer": optimizer.state_dict(), "random_state": random.getstate(),
                        "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all()},
                       checkpoint / "optimizer.pt")
            (checkpoint / "trainer_state.json").write_text(json.dumps({
                "algorithm": algorithm,
                "settings": _resume_settings(settings),
                "checkpoint_policy": {"save_steps": save_steps, "save_total_limit": save_total_limit},
                "seeds": str(Path(raw["seeds"]).resolve()),
                "seed_sha256": hashlib.sha256(Path(raw["seeds"]).read_bytes()).hexdigest(),
                "verifier_model": raw["verifier"].get("model"),
                "iteration": iteration,
                "mastery": curriculum.mastery,
                "metrics": metrics,
                "reference_checkpoint": str(sft_checkpoint),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            _prune_checkpoints(output, algorithm, save_total_limit,
                               protected=[sft_checkpoint, checkpoint] + ([Path(resume)] if resume else []))
        print(json.dumps({"iteration": iteration, "category": category, **metrics,
                          "checkpoint": str(checkpoint) if saved_this_iteration else None}, ensure_ascii=False), flush=True)
    return checkpoint
