# Train the Agentic Proposer

This repository has the following executable stages:

1. `collect-sft`: run the API Teacher in the existing closed-loop harness, execute tools, require an exact-candidate self-check, and retain only verifier-accepted trajectories.
2. `sft`: behavior-clone every observable assistant action in each accepted trajectory. The prompt is the corresponding observation, including current Skill metadata, stage, history, and tool schemas. The loss masks all prompt tokens.
3. `grpo`: online proposer GRPO with group-normalized terminal rewards, token PPO clipping and sampled reference KL. It shares the proposer rollout harness and frozen prober with MGPO, but does not score process rewards.
4. `mgpo`: sample groups of on-policy proposer trajectories against the same objective, verify submitted problems, estimate the fraction of 16 frozen-prober samples that answer correctly, and update the proposer using the paper's trajectory and stage advantages plus the asymmetric gate.
5. `solver-grpo`: train a separate downstream Solver on accepted final `{question, answer}` pairs from MGPO. Proposer traces, verifier votes, and prober outputs are excluded from this dataset.

## Run

On a machine with CUDA GPUs, from the repository root:

```bash
export ENVMUX_API_KEY='YOUR_KEY'
bash scripts/train_sft.sh --base-model /models/Qwen3-4B
bash scripts/train_grpo.sh --prober-model /models/Qwen3-4B
bash scripts/train_mgpo.sh --prober-model /models/Qwen3-4B
```

The scripts create `.venv-proposer` and install training dependencies. GRPO and MGPO are separate runs from the same SFT checkpoint. See [launcher options](../scripts/TRAINING_SCRIPTS.md) for dry runs, model paths and resume options. The bundled SFT dataset is the user-selected 136 accepted trajectories (1,543 actions); its explicit training minimum is 136. To collect API demonstrations on a machine without CUDA:

```bash
export ENVMUX_API_KEY='YOUR_KEY'
bash scripts/run_proposer_training.sh collect-sft
```

Move the repository and workspace to a CUDA machine, then run the remaining stages:

```bash
bash scripts/run_proposer_training.sh sft
bash scripts/run_proposer_training.sh mgpo
bash scripts/run_proposer_training.sh solver-grpo
```

The default [math configuration](../configs/proposer_training.math.json) uses `qwen3.8-flash` through EnvMux for the Teacher and verifier. The API key is read from `ENVMUX_API_KEY` or an ignored local credential file. The local Proposer and Solver start from `Qwen/Qwen3-4B-Instruct-2507`. Edit the JSON config and pass `--config path/to/config.json` to any stage to use other models or paths.

## Input data and Skill library

The seed file is JSONL, one objective per line:

```json
{"id":"algebra-1","category":"algebra","skill_ids":["math.algebra.equation-construction"],"objective":"Construct a novel, self-contained algebra problem with one verifiable scalar answer."}
```

The default math configuration ships 40 bounded math-construction Skills with exact calculators and 10 checking/revision Skills, plus 240 seed objectives. The GRPO/MGPO launchers select a separate 100-prompt file. These are project-authored assets, not the official paper corpus; see [math data and verification scope](MATH_PROPOSER.md). The optional `proposer_training.json` configuration retains a small synthetic demonstration library.

`collect-sft` writes one accepted trajectory per JSONL record. Each contains `trajectory_id`, `seed_id`, `category`, `objective`, `transitions`, `task_bundle`, and `valid`. Each transition has the exact observation messages, available tool schemas, stage, one observable assistant action, and actual tool results. Failed attempts with available traces are preserved in `sft_trajectories.failed.jsonl` with `valid:false` and a `failure_reason`; they are diagnostic data, not positive SFT examples. Rejection summaries remain in a separate file. API credentials are redacted and never written to these files.

## Objective mapping

The Agentic SFT target follows [paper Eq. 5](https://arxiv.org/html/2602.03279v1#S3.SS4). The MGPO code follows Eqs. 6–10:

- The category sampler uses inverse EMA mastery.
- The terminal reward is `V(q) * (R_base + 1[rho > 0] * lambda * (1-rho))`.
- Terminal rewards are standardized within a group of rollouts for the same objective. Nonnegative process rewards are standardized within the same cognitive stage. The fused advantage uses `omega=0.5`.
- Fused advantages and implicit rewards `beta * log(pi_theta/pi_ref)` are each centered within a stage group. The weight is their difference.
- Each action ratio `pi_theta/pi_old` passes through `sech²(tau*(ratio-1)/2)`, with `tau_pos=1.0` and `tau_neg=1.05`. The update is token-normalized weighted maximum likelihood. We hold weights fixed during each update.

The frozen prober uses 16 independent samples at temperature 1.4. Tool success supplies process reward; optional configured-verifier judgments supply a reward for coherent, observable reflections. Invalid final trajectories have their process rewards masked in advantage estimation.

The downstream Solver GRPO implementation follows the paper's Appendix 7.3: 3 epochs, learning rate `1e-6`, cosine schedule, group size 8, rollout temperature 0.7, maximum sequence length 20,480, KL coefficient 0.05, clipping 0.2, AdamW betas `(0.9, 0.95)`, and weight decay 0.1. The paper's global batch is 128; the pilot config uses 8 so it can consume a small dataset. Set `global_batch_size` to 128 for a full run. The implementation pins TRL 0.25.0 and selects its `grpo` loss type and group reward scaling.

## Scope of this reproduction

The paper does not publish SFT episode count, MGPO update count, process-reward scale, or proposer optimizer settings. Their values in the config are explicit engineering choices. The general math config uses 16 RL updates; the 100-prompt configuration uses 100 GRPO or MGPO updates; it is not the paper's 10,000–11,000 accepted-problem budget. Set a larger curriculum and `solver_grpo.require_paper_budget=true` after supplying a large validated Skill library and seed pool.

The paper uses three different verifier models and a prober pool. The math config uses the requested `qwen3.8-flash` adaptation: three privileged votes and one blind vote from the same model. Its voting accuracy should not be equated with the paper's reported ensemble accuracy. The built-in prober and Solver grader handle scalar non-code answers; coding tasks require a separate execution-based grader before their rewards can be used. Full 4B training also requires substantial CUDA memory and time. No official checkpoint, official corpus, or benchmark result is bundled here.
