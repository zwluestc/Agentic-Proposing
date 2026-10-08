#!/usr/bin/env python3
"""Check accepted SFT actions with the training tokenizer, without loading weights.

Run after installing .[training]. The tokenizer must be local or already cached;
this audit never downloads files and does not alter the dataset or configuration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_proposing.training.data import encode_sft_example, read_jsonl


class _ObservedTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []
        self.completion_tokens = 0

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def encode(self, text, **kwargs):
        tokens = self.tokenizer.encode(text, **kwargs)
        self.completion_tokens = len(tokens) + int(self.tokenizer.eos_token_id is not None)
        return tokens

    def apply_chat_template(self, messages, **kwargs):
        tokens = self.tokenizer.apply_chat_template(messages, **kwargs)
        self.calls.append((len(messages), len(tokens)))
        return tokens


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/proposer_training.math.json"))
    parser.add_argument("--tokenizer", help="Local model/tokenizer path or already cached model ID")
    args = parser.parse_args(argv)
    try:
        from transformers import AutoTokenizer

        config_path = Path(args.config)
        if not config_path.is_file():
            config_path = ROOT / config_path
        settings = json.loads(config_path.read_text(encoding="utf-8"))["sft"]
        data_path = Path(settings["data_file"])
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        rows = [row for row in read_jsonl(data_path) if row.get("valid") is True]
        if not rows or any(not row.get("transitions") for row in rows):
            raise ValueError("Accepted trajectories with nonempty transitions are required")
        model_id = args.tokenizer or settings["base_model"]
        tokenizer = _ObservedTokenizer(AutoTokenizer.from_pretrained(
            model_id, use_fast=True, local_files_only=True,
        ))
        limit = int(settings["max_length"])
        successes, trimmed, original_sizes, retained_sizes = 0, 0, [], []
        failures = []
        completion_sizes = []
        for row in rows:
            for index, transition in enumerate(row["transitions"], 1):
                tokenizer.calls = []
                tokenizer.completion_tokens = 0
                try:
                    sample = encode_sft_example(tokenizer, transition, limit)
                    successes += 1
                    trimmed += len(tokenizer.calls) > 1
                    retained_sizes.append(len(sample["input_ids"]))
                except ValueError as exc:
                    failures.append({
                        "trajectory_id": row["trajectory_id"], "action_index": index,
                        "reason": str(exc),
                    })
                completion_sizes.append(tokenizer.completion_tokens)
                if tokenizer.calls:
                    original_sizes.append(tokenizer.calls[0][1] + tokenizer.completion_tokens)
        report = {
            "tokenizer": model_id, "data_file": str(data_path), "max_length": limit,
            "trajectories": len(rows), "action_samples": successes + len(failures),
            "encoded_successfully": successes, "failed": len(failures),
            "without_history_removal": successes - trimmed, "with_history_removal": trimmed,
            "original_full_sequence_max_tokens": max(original_sizes, default=0),
            "retained_sequence_max_tokens": max(retained_sizes, default=0),
            "completion_max_tokens": max(completion_sizes, default=0),
            "failure_reasons": dict(Counter(item["reason"] for item in failures)),
            "failure_examples": failures[:10],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return int(bool(failures))
    except (ImportError, OSError, ValueError, KeyError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
