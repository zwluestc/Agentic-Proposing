#!/usr/bin/env python3
"""Build a reproducible 100-prompt pool for online proposer MGPO rollouts."""
import json
from collections import defaultdict
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[1]


def main():
    rng = random.Random(42)
    grouped = defaultdict(list)
    source = ROOT / "configs/proposer_seeds.math.jsonl"
    for line in source.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            grouped[row["category"]].append(row)
    categories = sorted(grouped)
    if len(categories) != 40:
        raise ValueError("Expected 40 math construction families")
    extra = set(rng.sample(categories, 20))
    rows = []
    for category in categories:
        count = 3 if category in extra else 2
        selected = rng.sample(grouped[category], count)
        for index, seed in enumerate(selected, 1):
            row = dict(seed, id=f"grpo-{category}-{index}", source_seed_id=seed["id"])
            for skill_id in row["skill_ids"]:
                if not (ROOT / "configs/math_skills/atomic" / skill_id / "manifest.yaml").is_file():
                    raise ValueError(f"Missing Skill: {skill_id}")
            rows.append(row)
    rng.shuffle(rows)
    assert len(rows) == 100
    assert len({r["id"] for r in rows}) == 100
    assert len({r["objective"] for r in rows}) == 100
    destination = ROOT / "configs/proposer_seeds.math.grpo100.jsonl"
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    config = json.loads((ROOT / "configs/proposer_training.math.json").read_text())
    config["seeds"] = str(destination.relative_to(ROOT))
    config["mgpo"]["iterations"] = 100
    config["mgpo"]["output_dir"] = "research-workspace/math-proposer/training/mgpo-grpo100"
    config["grpo"]["iterations"] = 100
    config["grpo"]["output_dir"] = "research-workspace/math-proposer/training/grpo-proposer100"
    config["solver_grpo"]["data_file"] = config["mgpo"]["output_dir"] + "/rollouts.jsonl"
    (ROOT / "configs/proposer_training.math.grpo100.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps({"seeds": len(rows), "categories": len(categories),
                      "categories_with_3_seeds": len(extra), "random_seed": 42}))


if __name__ == "__main__":
    main()
