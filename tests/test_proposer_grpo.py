from __future__ import annotations

import math
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentic_proposing.config import RuntimeConfig
from agentic_proposing.training.grpo_objective import terminal_advantages, token_objective
from agentic_proposing.training.local_policy import action_logprob, action_token_logprobs, LocalPolicyProvider
from agentic_proposing.training.rl import (
    _resume_settings, _validate_resume, _update_grpo_policy, _prune_checkpoints,
    _restore_rollout_log, _check_fresh_output, train_grpo, train_mgpo,
)


def test_terminal_advantages_have_group_scope_and_ignore_tied_rewards():
    assert terminal_advantages([0.0, 2.0]) == pytest.approx([-1.0, 1.0])
    assert terminal_advantages([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]
    values = terminal_advantages([0.0, 1.0, 2.0])
    assert sum(values) == pytest.approx(0.0)
    assert sum(value ** 2 for value in values) / 3 == pytest.approx(1.0)
    for invalid in ([1.0], [0.0, math.nan], [math.inf, 1.0]):
        with pytest.raises(ValueError):
            terminal_advantages(invalid)


def test_resume_rejects_algorithm_reference_and_training_setting_changes(tmp_path):
    checkpoint = tmp_path / "sft"
    settings = {"group_size": 8, "iterations": 10, "beta": 0.05, "output_dir": "out"}
    state = {"algorithm": "grpo", "reference_checkpoint": str(checkpoint),
             "settings": _resume_settings(settings)}
    _validate_resume(state, dict(settings, iterations=20, resume_from="ckpt"), "grpo", checkpoint)
    _validate_resume(state, dict(settings, save_steps=20, save_total_limit=3), "grpo", checkpoint)
    with pytest.raises(ValueError, match="Cannot resume"):
        _validate_resume(state, settings, "mgpo", checkpoint)
    with pytest.raises(ValueError, match="reference checkpoint"):
        _validate_resume(state, settings, "grpo", tmp_path / "other")
    with pytest.raises(ValueError, match="settings differ"):
        _validate_resume(state, dict(settings, beta=0.1), "grpo", checkpoint)
    # Existing MGPO checkpoints remain usable with their original reference.
    _validate_resume({"reference_checkpoint": str(checkpoint)}, settings, "mgpo", checkpoint)


@pytest.mark.parametrize("trainer", [train_grpo, train_mgpo])
def test_rl_fails_before_loading_models_when_launched_with_multiple_processes(monkeypatch, trainer):
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="one process"):
        trainer({}, RuntimeConfig())


def test_grpo_rejects_greedy_rollouts_before_loading_weights(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match="temperature"):
        train_grpo({"grpo": {"rollout_temperature": 0.0}}, RuntimeConfig())


def test_checkpoint_pruning_keeps_reference_resume_and_foreign_directories(tmp_path):
    for iteration in range(1, 7):
        path = tmp_path / f"checkpoint-{iteration:04d}"
        path.mkdir()
        (path / "trainer_state.json").write_text(json.dumps({"algorithm": "grpo", "iteration": iteration}))
    foreign = tmp_path / "checkpoint-0007"
    foreign.mkdir()
    (foreign / "trainer_state.json").write_text(json.dumps({"algorithm": "mgpo", "iteration": 7}))
    unknown = tmp_path / "checkpoint-0008"
    unknown.mkdir()
    reference = tmp_path / "checkpoint-0001" / "reference"
    reference.mkdir()
    resume = tmp_path / "checkpoint-0002"
    link = tmp_path / "checkpoint-0009"
    link.symlink_to(resume, target_is_directory=True)
    _prune_checkpoints(tmp_path, "grpo", 2, protected=[reference, resume])
    assert (tmp_path / "checkpoint-0001").exists()
    assert resume.exists()
    assert not (tmp_path / "checkpoint-0003").exists()
    assert not (tmp_path / "checkpoint-0004").exists()
    assert (tmp_path / "checkpoint-0005").exists()
    assert (tmp_path / "checkpoint-0006").exists()
    assert foreign.exists() and unknown.exists() and link.is_symlink()


def test_resume_archives_rollouts_newer_than_checkpoint(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    rows = [{"algorithm": "grpo", "iteration": i, "run_id": str(i)} for i in (1, 2, 3)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _restore_rollout_log(path, 2, "grpo")
    assert [json.loads(line)["iteration"] for line in path.read_text().splitlines()] == [1, 2]
    archive = tmp_path / "rollouts.superseded.jsonl"
    assert json.loads(archive.read_text())["iteration"] == 3
    _restore_rollout_log(path, 2, "grpo")
    assert len(archive.read_text().splitlines()) == 1


@pytest.mark.parametrize("trainer,algorithm", [(train_grpo, "grpo"), (train_mgpo, "mgpo")])
@pytest.mark.parametrize("existing", ["rollouts", "checkpoint"])
def test_fresh_training_rejects_existing_output_before_loading_models(tmp_path, monkeypatch, trainer, algorithm, existing):
    monkeypatch.setenv("WORLD_SIZE", "1")
    if existing == "rollouts":
        (tmp_path / "rollouts.jsonl").write_text('{}\n')
    else:
        # Even incomplete saves must not be silently overwritten.
        (tmp_path / "checkpoint-0001").mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError("Output guard must run before loading model weights")

    monkeypatch.setattr("agentic_proposing.training.rl.load_model", forbidden)
    with pytest.raises(ValueError, match="--resume-from or a new --output-dir"):
        trainer({algorithm: {"output_dir": str(tmp_path)}}, RuntimeConfig())


def test_fresh_output_allows_new_directory_or_empty_rollout_log(tmp_path):
    _check_fresh_output(tmp_path / "new")
    (tmp_path / "rollouts.jsonl").touch()
    _check_fresh_output(tmp_path)


def _tiny_policy():
    torch = pytest.importorskip("torch")

    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scores = torch.nn.Parameter(torch.zeros(3))
            self.config = SimpleNamespace(use_cache=False)
            self.generation_settings = {}

        @property
        def device(self):
            return self.scores.device

        def forward(self, input_ids, **kwargs):
            return SimpleNamespace(logits=self.scores.expand(*input_ids.shape, 3))

        def generate(self, **kwargs):
            self.generation_settings = kwargs
            return torch.cat((kwargs["input_ids"], torch.tensor([[1, 2]])), dim=1)

        def save_pretrained(self, path):
            torch.save(self.state_dict(), Path(path) / "weights.pt")

    return TinyPolicy()


def test_token_scoring_excludes_prompt_and_matches_joint_logprob():
    torch = pytest.importorskip("torch")
    model = _tiny_policy()
    with torch.no_grad():
        model.scores.copy_(torch.tensor([0.1, 0.4, -0.2]))
    tokens = action_token_logprobs(model, [0, 1, 0], [1, 2], temperature=0.7)
    expected = torch.log_softmax(model.scores / 0.7, dim=0)[torch.tensor([1, 2])]
    assert torch.allclose(tokens, expected)
    assert torch.allclose(action_logprob(model, [0], [1, 2]), action_token_logprobs(model, [0], [1, 2]).sum())
    tokens.sum().backward()
    assert model.scores.grad is not None


def test_clipping_stops_overrewarded_and_overpenalized_token_gradients():
    torch = pytest.importorskip("torch")
    # Ratios of 1.5 with positive advantage and 0.5 with negative advantage clip.
    for ratio, advantage in [(1.5, 1.0), (0.5, -1.0)]:
        current = torch.tensor([math.log(ratio) - 2], requires_grad=True)
        old = torch.tensor([-2.0], requires_grad=True)
        reference = torch.tensor([-2.0], requires_grad=True)
        losses, kl, clipped = token_objective(current, old, reference, advantage, beta=0.0)
        losses.sum().backward()
        assert current.grad.item() == pytest.approx(0.0)
        assert clipped.item() == 1.0
        assert kl.item() >= 0
        assert old.grad is None and reference.grad is None


def test_reference_kl_penalty_pulls_current_logprob_toward_reference():
    torch = pytest.importorskip("torch")
    current = torch.tensor([-1.0], requires_grad=True)
    old = torch.tensor([-1.0])
    reference = torch.tensor([-2.0])
    losses, kl, _ = token_objective(current, old, reference, 0.0, beta=0.5)
    losses.sum().backward()
    assert kl.item() == pytest.approx(math.exp(-1) - 1 + 1)
    assert current.grad.item() > 0  # Gradient descent reduces the overlarge log prob.


def _action(model, completion):
    logs = action_token_logprobs(model, [0], completion).detach().tolist()
    return {"process_reward": 999.0, "stage": "anything", "policy_raw": {
        "prompt_token_ids": [0], "completion_token_ids": completion,
        "old_token_logprobs": logs, "reference_token_logprobs": logs,
        "logprob_temperature": 1.0,
    }}


def test_grpo_update_uses_terminal_reward_and_is_invariant_to_turn_splits():
    torch = pytest.importorskip("torch")
    final_parameters = []
    for split in (False, True):
        model = _tiny_policy()
        positive = [_action(model, [1]), _action(model, [1])] if split else [_action(model, [1, 1])]
        group = [{"terminal_reward": 2.0, "transitions": positive},
                 {"terminal_reward": 0.0, "transitions": [_action(model, [2])]}]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        metrics = _update_grpo_policy(model, optimizer, group, {"beta": 0.0})
        assert model.scores[1].item() > model.scores[2].item()
        assert metrics["tokens"] == 3
        assert metrics["clip_fraction"] == 0
        final_parameters.append(model.scores.detach())
    assert torch.allclose(*final_parameters)


def test_provider_grpo_sampling_and_scoring_share_temperature(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr("agentic_proposing.training.local_policy.encode_prompt", lambda *args: [0, 0])
    model = _tiny_policy()
    tokenizer = SimpleNamespace(eos_token_id=2, decode=lambda *args, **kwargs: '{"content":"done","tool_calls":[]}')
    provider = LocalPolicyProvider(model, tokenizer, reference=model, record_token_logprobs=True)
    turn = provider.complete([], [], temperature=0.7)
    assert model.generation_settings["top_p"] == 1.0
    assert model.generation_settings["top_k"] == 0
    assert turn.raw["logprob_temperature"] == 0.7
    assert len(turn.raw["old_token_logprobs"]) == len(turn.raw["completion_token_ids"])
    assert turn.raw["old_token_logprobs"] == turn.raw["reference_token_logprobs"]


def test_grpo_loop_and_resume_save_real_optimizer_without_process_judging(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import agentic_proposing.training.rl as rl

    monkeypatch.setenv("WORLD_SIZE", "1")
    sft = tmp_path / "sft" / "final"
    sft.mkdir(parents=True)
    seeds = tmp_path / "seeds.jsonl"
    seeds.write_text(json.dumps({"id": "one", "category": "math", "objective": "Construct a task"}) + "\n")
    tokenizer = SimpleNamespace(save_pretrained=lambda path: (Path(path) / "tokenizer.json").write_text("{}"))

    def load(source, *, trainable):
        model = _tiny_policy()
        weights = Path(source) / "weights.pt"
        if weights.exists():
            model.load_state_dict(torch.load(weights, weights_only=True))
        return model, tokenizer

    monkeypatch.setattr(rl, "load_model", load)
    monkeypatch.setattr("agentic_proposing.training.runner._prepare_workspace", lambda raw: tmp_path)
    monkeypatch.setattr(rl, "seed_run_state", lambda *args: None)
    monkeypatch.setattr(rl, "create_provider", lambda raw: object())
    monkeypatch.setattr(rl, "FrozenProber", lambda *args, **kwargs: SimpleNamespace(pass_rate=lambda bundle: 0.5))

    def forbidden(*args, **kwargs):
        raise AssertionError("Terminal-only GRPO must not invoke process reward judging")

    monkeypatch.setattr(rl, "_process_reward", forbidden)

    def runtime(workspace, actor, config, *, verifier_provider, action_observer):
        import itertools
        members = itertools.cycle([True, False])

        def run(*args):
            valid = next(members)
            action_observer(_action(actor.model, [1 if valid else 2]))
            return SimpleNamespace(status="completed" if valid else "max_turns", run_id=str(valid),
                                   state=SimpleNamespace(working_artifact={"answer": "1"}))

        return SimpleNamespace(harness=SimpleNamespace(run=run), registry=None)

    monkeypatch.setattr(rl, "build_runtime", runtime)
    raw = {"workspace": str(tmp_path), "seeds": str(seeds), "verifier": {"model": "test-only"},
           "sft": {"output_dir": str(sft.parent), "base_model": "test-only"},
           "grpo": {"output_dir": str(tmp_path / "rl"), "iterations": 1, "group_size": 2, "beta": 0.0}}
    first = train_grpo(raw, RuntimeConfig())
    first_state = json.loads((first / "trainer_state.json").read_text())
    assert first_state["algorithm"] == "grpo"
    assert first_state["reference_checkpoint"] == str(sft)
    assert (first / "optimizer.pt").is_file()
    raw["grpo"].update(iterations=2, resume_from=str(first))
    second = train_grpo(raw, RuntimeConfig())
    assert second.name == "checkpoint-0002"
    records = [json.loads(line) for line in (second.parent / "rollouts.jsonl").read_text().splitlines()]
    assert len(records) == 4
    assert [record["iteration"] for record in records] == [1, 1, 2, 2]
    assert all(row["transitions"][0]["process_reward"] == 0.0 for row in records)
    # Save only every other update and the final update; retain two recent saves.
    fresh = tmp_path / "periodic"
    raw["grpo"] = {"output_dir": str(fresh), "iterations": 5, "group_size": 2,
                   "beta": 0.0, "save_steps": 2, "save_total_limit": 2}
    final = train_grpo(raw, RuntimeConfig())
    assert final.name == "checkpoint-0005"
    assert sorted(path.name for path in fresh.glob("checkpoint-*")) == ["checkpoint-0004", "checkpoint-0005"]
    assert len((fresh / "rollouts.jsonl").read_text().splitlines()) == 10
