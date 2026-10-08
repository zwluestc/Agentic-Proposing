# Mapping to Agentic Proposing

Paper: [Agentic Proposing: Enhancing Large Language Model Reasoning via Compositional Skill Synthesis](https://arxiv.org/abs/2602.03279)

| Paper element | Reproduction |
|---|---|
| Static/open-loop vs dynamic/closed-loop Figure 1 | `AgentHarness` plus mandatory `submit_artifact` gates |
| `o_t = <K_t, h_t, sigma_t>` | `RunState.active_skill_ids`, event/transcript state, `RunState.stage` |
| Cognitive action | Structured model decision or `working_state_edit.decision` |
| `tau_exec` | General `ToolCall`: retrieval, Python, Skill verifier, web, parser, gates |
| `tau_edit` removes a Skill | `skill_prune` edits the active run graph; compose/stage/promote/deprecate/version operations manage persistent packages |
| nonterminal check/refine before `tau_submit` | `self_check_artifact(TaskBundle)` with verifier report and exact-candidate fingerprint |
| `tau_submit` | `submit_artifact(TaskBundle)`, optionally rejecting missing or stale self-checks |
| Structured Q&A acquisition | `SkillExtractor.extract_from_qa`, one item per extraction call |
| Unstructured corpus acquisition | document parser, semantic chunker, Evidence Store, `extract_from_corpus` |
| `k=<intent, method, difficulty, tool hint>` | discovery manifest + procedural `SKILL.md` + declared tools |
| Progressive disclosure | registry indexes `manifest.yaml`; `SKILL.md` is read only by `skill_load`/context source |
| Filesystem Skill package | `skills/{atomic,composite}/<id>/...` |
| Draft/Check/Refine/Finalize | functional stages and failed submit feedback loop |
| Dynamic pruning | active Skill list may be rewritten; incompatible composition is rejected |
| Verifier privilege | deterministic privileged gates plus optional isolated Teacher votes, a blind solver, majority agreement, and second audit |
| Prober | intentionally outside the minimal core; can be added as another verification tool |
| Agentic SFT | `training.runner collect-sft` plus `sft`: accepted observable tool trajectories and masked-token behavioral cloning |
| MGPO | `training.rl` and `training.mgpo_objective`: curriculum, layered rewards, group/stage advantages, implicit reward, asymmetric gate, token-normalized update |
| Downstream Solver GRPO | `training.solver_grpo`: final question/answer pairs only, Appendix 7.3 hyperparameters |

## Deliberate extensions

The paper composes Skills during proposing but does not fully specify recursive persistence of every successful composition. This codebase makes it explicit:

1. A composition is compiled into a first-class versioned Skill package.
2. It remains staged while independent Task instantiations are submitted through the full verifier pipeline.
3. Passing the replay threshold promotes the composite; only then can it be selected as a parent in later DAGs.
4. Accepted tasks plus observable trajectories are reverse-mined for emergent Skills.
5. Emergent candidates remain staged and require deduplication, replay evidence, and explicit promotion.

The paper's code tool is generalized into a typed Tool protocol. Workspace inspection, textbook parsing, local evidence retrieval, configurable web search, Web-to-Evidence ingestion, isolated calculations, persistent analysis sessions, trusted Skill scripts, contamination checks, dynamic external providers, and Skill Graph operations all use the same dispatch/hook/event/permission mechanism. These harness extensions generalize Figure 1's executable action without changing the paper's proposing objective.
