from __future__ import annotations

from pathlib import Path
from typing import Any
import json
import os

from .data import read_jsonl
from .prober import answers_match, extract_boxed


def train_solver_grpo(raw: dict[str, Any]) -> Path:
    """Paper Appendix 7.3: GRPO on accepted final question/answer pairs only."""
    try:
        import torch
        from datasets import Dataset
        from transformers import AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError('Install training dependencies: python3 -m pip install -e ".[training]"') from exc
    if not torch.cuda.is_available():
        raise RuntimeError("Solver GRPO requires CUDA")
    settings = dict(raw["solver_grpo"])
    trajectories = read_jsonl(settings.get(
        "data_file", str(Path(raw["mgpo"]["output_dir"]) / "rollouts.jsonl")
    ))
    rows = []
    seen_questions: set[str] = set()
    for trajectory in trajectories:
        if not trajectory.get("valid"):
            continue
        task = dict(trajectory["task_bundle"])
        if isinstance(task.get("answer"), (dict, list)) or task.get("task_type") in {"coding", "code", "programming"}:
            raise ValueError("Built-in Solver GRPO grader supports scalar non-code answers only")
        question = str(task["question"]).strip()
        gold = str(task["answer"])
        accepted_answers = [gold]
        if task.get("task_type") == "single_choice":
            options = task.get("metadata", {}).get("options", [])
            if isinstance(options, list) and options:
                question += "\nOptions: " + "; ".join(
                    f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
                )
                if len(gold.strip()) == 1 and gold.strip().upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(options)]:
                    accepted_answers.append(str(options[ord(gold.strip().upper()) - 65]))
                else:
                    accepted_answers.extend(
                        chr(65 + index) for index, option in enumerate(options)
                        if answers_match(option, gold)
                    )
        if question in seen_questions:
            continue
        seen_questions.add(question)
        rows.append({
            "prompt": [
                {"role": "system", "content": "Solve the problem. Put exactly one final answer in \\boxed{...}."},
                {"role": "user", "content": question},
            ],
            "accepted_answers": accepted_answers,
        })
    if not rows:
        raise ValueError("No accepted final question/answer pairs for Solver GRPO")
    if len(rows) < int(settings.get("group_size", 8)):
        raise ValueError("Solver GRPO needs at least one group of distinct accepted problems")
    if bool(settings.get("require_paper_budget", False)) and len(rows) < int(settings.get("paper_budget", 10000)):
        raise ValueError("Insufficient accepted pairs for the configured paper budget")
    dataset = Dataset.from_list(rows)

    def binary_outcome_reward(completions, accepted_answers, **kwargs):
        rewards = []
        for completion, gold_values in zip(completions, accepted_answers):
            text = completion if isinstance(completion, str) else "".join(
                str(message.get("content", "")) for message in completion
            )
            boxed = extract_boxed(text)
            rewards.append(float(boxed is not None and any(
                answers_match(boxed, gold) for gold in gold_values
            )))
        return rewards

    model_id = str(settings.get("model", raw["sft"]["base_model"]))
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    per_device = int(settings.get("per_device_batch_size", 2))
    global_batch = int(settings.get("global_batch_size", 128))
    if global_batch % (world_size * per_device):
        raise ValueError("global_batch_size must be divisible by WORLD_SIZE * per_device_batch_size")
    max_sequence = int(settings.get("max_sequence_length", 20480))
    max_prompt = int(settings.get("max_prompt_length", 2048))
    if max_prompt >= max_sequence:
        raise ValueError("max_prompt_length must be below max_sequence_length")
    destination = Path(settings["output_dir"]).resolve()
    args = GRPOConfig(
        output_dir=str(destination),
        num_train_epochs=float(settings.get("epochs", 3)),
        learning_rate=float(settings.get("learning_rate", 1e-6)),
        lr_scheduler_type="cosine",
        num_generations=int(settings.get("group_size", 8)),
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=global_batch // (world_size * per_device),
        temperature=float(settings.get("temperature", 0.7)),
        max_prompt_length=max_prompt,
        max_completion_length=max_sequence - max_prompt,
        beta=float(settings.get("beta", 0.05)),
        epsilon=float(settings.get("clip_epsilon", 0.2)),
        adam_beta1=0.9,
        adam_beta2=0.95,
        weight_decay=float(settings.get("weight_decay", 0.1)),
        scale_rewards="group",
        loss_type="grpo",
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        save_strategy="epoch",
        model_init_kwargs={"torch_dtype": "bfloat16", "attn_implementation": "sdpa"},
    )
    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=binary_outcome_reward,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()
    final = destination / "final"
    trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final)
        print(json.dumps({"solver_grpo_checkpoint": str(final), "pairs": len(rows)}))
    return final
