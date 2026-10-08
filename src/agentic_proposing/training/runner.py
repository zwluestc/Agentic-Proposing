from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import os
import sys
import shutil

from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.config import RuntimeConfig, create_provider, load_mapping, resolve_provider_config
from agentic_proposing.runtime import build_runtime
from agentic_proposing.skills.registry import SkillRegistry
from agentic_proposing.core.store import WorkspaceLayout
from .data import encode_sft_example, read_jsonl, seed_objective, seed_run_state, write_jsonl


def _runtime_config(raw: dict[str, Any]) -> RuntimeConfig:
    settings = dict(raw.get("runtime", {}))
    settings.setdefault("harness", {"max_turns": 20, "temperature": 0.5, "max_tokens_per_turn": 2048})
    settings.setdefault("verification", {
        "require_self_check": True,
        "model_committee_enabled": True,
        "model_voters": 3,
        "second_audit": True,
    })
    settings.setdefault("permissions", {
        "allow_read_only": True,
        "allow_registry_writes": False,
        "allow_network": False,
        "allow_process": False,
    })
    settings.setdefault("skill_mining", {"enabled": False})
    return RuntimeConfig(**settings)


def _prepare_workspace(raw: dict[str, Any]) -> Path:
    workspace = Path(raw["workspace"]).resolve()
    layout = WorkspaceLayout(workspace)
    layout.initialize()
    if raw.get("skill_library"):
        source = Path(raw["skill_library"]).resolve()
        for manifest in source.glob("*/manifest.yaml"):
            destination = layout.skills / "atomic" / manifest.parent.name
            if destination.exists():
                if (destination / "manifest.yaml").read_bytes() != manifest.read_bytes():
                    raise ValueError(f"Skill differs from supplied library: {destination}")
            else:
                shutil.copytree(manifest.parent, destination)
    if not SkillRegistry(layout).discovery_summaries() and raw.get("bootstrap_synthetic_skills", False):
        bootstrap_demo_workspace(workspace)
    if not SkillRegistry(layout).discovery_summaries():
        raise ValueError(
            "No verified Skills in the workspace. Promote replay-validated Skills, "
            "or set bootstrap_synthetic_skills=true for a synthetic demonstration."
        )
    return workspace


def collect_sft(raw: dict[str, Any]) -> Path:
    if raw.get("sft", {}).get("target_accepted") is not None:
        from .collection import collect_to_target
        return collect_to_target(raw)
    for role in ("teacher", "verifier"):
        provider = resolve_provider_config(raw[role])
        env_name = str(provider.get("api_key_env", "OPENAI_API_KEY"))
        if not provider.get("api_key") and not os.environ.get(env_name):
            raise RuntimeError(f"Set {env_name} before calling the {role} API")
    workspace = _prepare_workspace(raw)
    settings = dict(raw["sft"])
    output = Path(settings["data_file"]).resolve()
    seeds = read_jsonl(raw["seeds"])
    teacher = create_provider(dict(raw["teacher"]))
    verifier = create_provider(dict(raw["verifier"]))
    records: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def observe(transition: dict[str, Any]) -> None:
        current.append(transition)

    runtime = build_runtime(
        workspace, teacher, _runtime_config(raw),
        verifier_provider=verifier, action_observer=observe,
    )
    accepted_questions: set[str] = set()
    rejected: list[dict[str, Any]] = []
    for index in range(int(settings.get("episodes", 16))):
        seed = seeds[index % len(seeds)]
        current.clear()
        objective = seed_objective(seed) + f" Instance index: {index + 1}."
        result = runtime.harness.run(
            objective, seed_run_state(seed, objective, workspace, runtime.registry)
        )
        bundle = dict(result.state.working_artifact)
        question = str(bundle.get("question", "")).strip()
        if result.status == "completed" and question and question not in accepted_questions:
            accepted_questions.add(question)
            records.append({
                "trajectory_id": result.run_id,
                "seed_id": str(seed.get("id", index)),
                "category": str(seed.get("category", "general")),
                "objective": objective,
                "transitions": list(current),
                "task_bundle": bundle,
                "valid": True,
            })
            print(f"accepted {len(records)} / {index + 1}: {result.run_id}", flush=True)
        else:
            rejected.append({"run_id": result.run_id, "status": result.status,
                             "duplicate": bool(question in accepted_questions)})
            print(f"rejected {index + 1}: {result.status}", flush=True)
    write_jsonl(output, records)
    write_jsonl(output.with_suffix(".rejected.jsonl"), rejected)
    if not records:
        raise RuntimeError("The Teacher produced no verifier-accepted SFT trajectories")
    print(json.dumps({"sft_data": str(output), "accepted": len(records),
                      "rejected": len(rejected)}, ensure_ascii=False))
    return output


def accepted_sft_trajectories(settings: dict[str, Any]) -> list[dict[str, Any]]:
    trajectories = [row for row in read_jsonl(settings["data_file"]) if row.get("valid") is True]
    minimum = int(settings.get("min_accepted_for_training", settings.get("target_accepted", 1)))
    if minimum < 1:
        raise ValueError("min_accepted_for_training must be positive")
    if len(trajectories) < minimum:
        raise ValueError(f"SFT requires {minimum} accepted trajectories; found {len(trajectories)}")
    if any(not row.get("transitions") for row in trajectories):
        raise ValueError("Accepted SFT trajectories must contain actions")
    return trajectories


def training_plan(raw: dict[str, Any], stage: str) -> dict[str, Any]:
    """Inspect public inputs without loading weights or invoking APIs."""
    if stage == "sft":
        settings = raw["sft"]
        rows = accepted_sft_trajectories(settings)
        return {
            "stage": stage, "base_model": settings["base_model"],
            "data_file": settings["data_file"], "trajectories": len(rows),
            "action_samples": sum(len(row["transitions"]) for row in rows),
            "max_length": settings.get("max_length", 8192),
            "output_dir": settings["output_dir"],
        }
    if stage in {"grpo", "mgpo"}:
        settings = raw[stage]
        seeds = read_jsonl(raw["seeds"])
        if not seeds:
            raise ValueError("RL requires at least one seed prompt")
        if any(not seed.get("objective") for seed in seeds):
            raise ValueError("Every RL seed must contain an objective")
        if int(settings["group_size"]) < 2 or int(settings["iterations"]) < 1:
            raise ValueError("RL requires group_size >= 2 and iterations >= 1")
        checkpoint = Path(raw["sft"].get("checkpoint", str(Path(raw["sft"]["output_dir"]) / "final")))
        return {
            "stage": stage, "seeds": raw["seeds"], "prompt_count": len(seeds),
            "sft_checkpoint": str(checkpoint), "sft_checkpoint_exists": checkpoint.is_dir(),
            "prober_model": settings.get("prober_model", raw["sft"]["base_model"]),
            "verifier_model": raw["verifier"]["model"],
            "iterations": settings["iterations"], "group_size": settings["group_size"],
            "max_context_tokens": settings.get("max_context_tokens", 8192),
            "output_dir": settings["output_dir"], "resume_from": settings.get("resume_from"),
        }
    raise ValueError("--dry-run supports sft, grpo and mgpo")


def train_sft(raw: dict[str, Any]) -> Path:
    settings = dict(raw["sft"])
    trajectories = accepted_sft_trajectories(settings)
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import Trainer, TrainingArguments
    except ImportError as exc:
        raise RuntimeError('Install training dependencies: python3 -m pip install -e ".[training]"') from exc
    from .local_policy import load_model

    model, tokenizer = load_model(str(settings["base_model"]), trainable=True)
    examples = [
        encode_sft_example(tokenizer, transition, int(settings.get("max_length", 8192)))
        for trajectory in trajectories if trajectory.get("valid") is True
        for transition in trajectory["transitions"]
    ]
    if not examples:
        raise ValueError("SFT requires accepted trajectories with at least one action")

    class ActionDataset(Dataset):
        def __len__(self):
            return len(examples)

        def __getitem__(self, index):
            return examples[index]

    def collate(rows):
        length = max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor([
                row["input_ids"] + [tokenizer.pad_token_id] * (length - len(row["input_ids"]))
                for row in rows
            ], dtype=torch.long),
            "attention_mask": torch.tensor([
                row["attention_mask"] + [0] * (length - len(row["attention_mask"]))
                for row in rows
            ], dtype=torch.long),
            "labels": torch.tensor([
                row["labels"] + [-100] * (length - len(row["labels"]))
                for row in rows
            ], dtype=torch.long),
        }

    destination = Path(settings["output_dir"]).resolve()
    arguments = TrainingArguments(
        output_dir=str(destination),
        per_device_train_batch_size=int(settings.get("per_device_batch_size", 1)),
        gradient_accumulation_steps=int(settings.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(settings.get("epochs", 2)),
        learning_rate=float(settings.get("learning_rate", 2e-5)),
        lr_scheduler_type="cosine",
        warmup_ratio=float(settings.get("warmup_ratio", 0.03)),
        weight_decay=float(settings.get("weight_decay", 0.01)),
        bf16=True,
        logging_steps=int(settings.get("logging_steps", 10)),
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=ActionDataset(), data_collator=collate)
    trainer.train()
    final = destination / "final"
    trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final)
        print(json.dumps({"sft_checkpoint": str(final), "actions": len(examples)}))
    return final


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Agentic Proposer training pipeline")
    parser.add_argument("stage", choices=["collect-sft", "sft", "grpo", "mgpo", "solver-grpo", "all"])
    parser.add_argument("--config", default="configs/proposer_training.math.json")
    parser.add_argument("--base-model", help="Local path or Hugging Face ID for SFT")
    parser.add_argument("--sft-checkpoint", help="Explicit proposer checkpoint for GRPO/MGPO")
    parser.add_argument("--prober-model", help="Frozen prober model path or ID for GRPO/MGPO")
    parser.add_argument("--output-dir", help="Override this training stage's output directory")
    parser.add_argument("--iterations", type=int, help="Number of GRPO/MGPO updates")
    parser.add_argument("--resume-from", help="GRPO/MGPO checkpoint directory to resume")
    parser.add_argument("--dry-run", action="store_true", help="Inspect inputs; do not load models or call APIs")
    args = parser.parse_args(argv)
    try:
        raw = load_mapping(args.config)
        if args.base_model:
            raw["sft"]["base_model"] = args.base_model
        if args.sft_checkpoint:
            raw["sft"]["checkpoint"] = args.sft_checkpoint
        if args.output_dir:
            if args.stage not in {"sft", "grpo", "mgpo", "solver-grpo"}:
                raise ValueError("--output-dir requires a single training stage")
            raw[args.stage.replace("-", "_")]["output_dir"] = args.output_dir
        if args.prober_model or args.iterations is not None or args.resume_from:
            if args.stage not in {"grpo", "mgpo"}:
                raise ValueError("--prober-model, --iterations and --resume-from require grpo or mgpo")
            if args.prober_model:
                raw[args.stage]["prober_model"] = args.prober_model
            if args.iterations is not None:
                raw[args.stage]["iterations"] = args.iterations
            if args.resume_from:
                raw[args.stage]["resume_from"] = args.resume_from
        if args.dry_run:
            print(json.dumps(training_plan(raw, args.stage), ensure_ascii=False, indent=2))
            return 0
        if args.stage in {"grpo", "mgpo"}:
            plan = training_plan(raw, args.stage)
            if not plan["sft_checkpoint_exists"]:
                raise FileNotFoundError(f"SFT checkpoint missing: {plan['sft_checkpoint']}")
            if not resolve_provider_config(raw["verifier"]).get("api_key"):
                raise RuntimeError("Set the configured verifier API key before RL training")
        if args.stage == "all":
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError('Install training dependencies: python3 -m pip install -e ".[training]"') from exc
            if not torch.cuda.is_available():
                raise RuntimeError("The all-stage run requires CUDA; run collect-sft alone on this computer")
        if args.stage in {"collect-sft", "all"}:
            collect_sft(raw)
        if args.stage in {"sft", "all"}:
            train_sft(raw)
        if args.stage == "grpo":
            from .rl import train_grpo
            train_grpo(raw, _runtime_config(raw))
        if args.stage in {"mgpo", "all"}:
            from .rl import train_mgpo
            train_mgpo(raw, _runtime_config(raw))
        if args.stage in {"solver-grpo", "all"}:
            from .solver_grpo import train_solver_grpo
            train_solver_grpo(raw)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
