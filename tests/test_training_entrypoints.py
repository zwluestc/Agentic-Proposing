import json
from pathlib import Path

import pytest

from agentic_proposing.training.runner import accepted_sft_trajectories, main


def test_explicit_sft_training_minimum_allows_user_selected_dataset(tmp_path):
    data = tmp_path / "sft.jsonl"
    data.write_text(json.dumps({"valid": True, "transitions": [{"action": {}}]}) + "\n")
    settings = {"data_file": str(data), "target_accepted": 500}
    with pytest.raises(ValueError, match="requires 500"):
        accepted_sft_trajectories(settings)
    assert len(accepted_sft_trajectories(dict(settings, min_accepted_for_training=1))) == 1


def test_failed_or_empty_trajectories_cannot_satisfy_sft_minimum(tmp_path):
    data = tmp_path / "sft.jsonl"
    data.write_text(json.dumps({"valid": False, "transitions": [{"action": {}}]}) + "\n")
    settings = {"data_file": str(data), "min_accepted_for_training": 1}
    with pytest.raises(ValueError, match="found 0"):
        accepted_sft_trajectories(settings)
    data.write_text(json.dumps({"valid": True, "transitions": []}) + "\n")
    with pytest.raises(ValueError, match="must contain actions"):
        accepted_sft_trajectories(settings)


@pytest.mark.parametrize("stage", ["grpo", "mgpo"])
def test_rl_dry_run_reports_effective_inputs_without_models_or_credentials(tmp_path, capsys, stage):
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(json.dumps({"id": "example", "objective": "Construct one math problem."}) + "\n")
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "seeds": str(seeds),
        "sft": {"base_model": "example/base", "output_dir": str(tmp_path / "sft")},
        "verifier": {"model": "qwen3.8-flash", "api_key": "must-not-be-printed"},
        stage: {"group_size": 8, "iterations": 100, "output_dir": str(tmp_path / stage)},
    }))
    rc = main([stage, "--config", str(config), "--dry-run", "--iterations", "2",
               "--sft-checkpoint", str(tmp_path / "custom-checkpoint"),
               "--prober-model", "/models/prober", "--output-dir", str(tmp_path / "custom-output")])
    assert rc == 0
    captured = capsys.readouterr()
    assert "must-not-be-printed" not in captured.out
    plan = json.loads(captured.out)
    assert plan["iterations"] == 2 and plan["prompt_count"] == 1
    assert plan["sft_checkpoint_exists"] is False
    assert Path(plan["sft_checkpoint"]).name == "custom-checkpoint"
    assert plan["prober_model"] == "/models/prober"
    assert Path(plan["output_dir"]).name == "custom-output"
