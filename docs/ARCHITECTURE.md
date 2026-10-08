# Architecture

## Scope

This is a codebase-level reproduction of Agentic Proposing. A frontier model can supply the Teacher, Composer, Extractor, and optional Verifier policies. The repository supplies the harness, filesystem environment, tools, gates, provenance, recursion, and optional Agentic SFT/MGPO training path. The post-trained local model supplies the Proposer policy during MGPO rollout; see [TRAINING.md](TRAINING.md).

## Two loops

### Inner proposing loop

```text
Observation(active skills, event history, functional stage, workspace state)
  -> Teacher decision
  -> zero or more tool calls
  -> hook and permission checks
  -> tool execution
  -> structured results
  -> state/checkpoint update
  -> repeat
```

The functional stages retain the paper's `Draft -> Check -> Refine -> Finalize`, with discovery, composition, mining, and promotion around them. They are context indicators, not a rigid finite-state machine.

Plain text never terminates a successful run. `self_check_artifact` is a nonterminal deterministic probe: failure moves the run to `refine`, while success stores a fingerprint of the exact candidate. When freshness enforcement is enabled, any edit requires another self-check. The Teacher must call `submit_artifact`, which reruns the configured submission gates; rejection remains an ordinary observation and acceptance is terminal.

### Outer recursive evolution loop

```text
seed benchmark/corpus -> atomic candidates -> validate/promote -> K(t)
K(t) -> typed composition -> composite candidate C
C + evidence -> accepted TaskBundle N
N + observable trajectory + verification -> emergent candidate E
replay/dedup/validate(C, E) -> K(t+1)
```

`A + B -> C`, `C + A -> D`, and `Task N -> E` are all first-class derivations. A repeated skill type is legal when occurrences have different instance IDs and bindings. Stored compositions are DAGs; cycles are rejected. A newly compiled C remains staged until its configurable independent task-instantiation replays meet the success threshold; only then is it promoted and visible to the next composition round.

## Harness components

- `AgentHarness`: model/tool turn loop, terminal semantics, limits, checkpoints.
- `ContextManager`: metadata-first discovery, on-demand Skill bodies, deferred tool schemas, transcript compaction.
- `ToolRegistry` and `ToolDispatcher`: JSON-schema tools, deferred discovery, parallel read-only calls, sequential mutating calls, dynamic upsert/removal, centralized long-output spill, and errors returned to the model.
- `HookBus` and `PermissionPolicy`: pre/post model, pre/post tool, failure, compaction, submit, and skill lifecycle interception with deny/ask/allow decisions.
- `RunStore`: `state.json`, `events.jsonl`, tool outputs, artifacts, final bundles, and turn checkpoints.
- `SkillRegistry`: indexes only manifests; loads `SKILL.md` lazily; stages before promotion; archives deprecated and replaced versions.
- `EvidenceStore`: hash-deduplicated documents, exact chunks, FTS retrieval, locators, and URL/time/hash provenance for acquired pages.
- Workspace tools: bounded Read/Glob/Grep over configured roots and managed per-run artifact writes.
- Python session manager: per-run persistent restricted workers, import policy, timeouts, and guaranteed session-end cleanup.
- Web layer: configurable Brave, Tavily, SearXNG, or generic JSON search plus SSRF-resistant public fetching and Web-to-Evidence ingestion.
- External tool providers: transactional JSON-over-HTTP catalog refresh using the same permission, hook, discovery, and output path as built-ins.
- `VerificationPipeline`: schema, rubric, evidence, ground truth, lineage, output constraints, contamination, and an optional model-committee gate.
- `VerifierCommittee`: isolated privileged votes, one blind solve, majority/canonical-answer agreement, and an optional second audit.
- `RecursiveEvolutionEngine`: SOTA-Teacher composition, independent instantiation replays, thresholded promotion, and recursive task/Skill closure without RL.

## Workspace

```text
workspace/
  config.json
  sources/
  evidence/
    evidence.sqlite
    documents/
    chunks/
  skills/
    atomic/<skill-id>/
    composite/<skill-id>/
    deprecated/
  staging/
    skills/<skill-id>@<version>/
    tasks/
  runs/<run-id>/
    state.json
    events.jsonl
    checkpoints/
    tool_outputs/
    artifacts/
    python_sessions/
    final/
```

## Context management

At session start the model receives compact Skill discovery summaries, not all procedures. `skill_load` adds selected bodies to future turns and enables their declared tools. Deferred tools are found with `tool_search`. Large outputs are written to artifacts and only bounded summaries return to the model. When transcript size crosses the configured budget, older messages are replaced by a structured state summary while the event log remains lossless on disk.

Permissions use `deny -> ask -> explicit allow -> side-effect class -> read-only default`. An embedding application may resolve `ask` callbacks; unattended runs fail closed. Read-only network or process tools cannot bypass their side-effect policy. Public Web fetch rejects local/private/reserved targets before the request and after redirects.

This deliberately separates:

- persistent rules and harness contract;
- mutable run state;
- event history;
- Skill procedures;
- source evidence;
- large tool artifacts.

## Observable traces

The implementation does not require or store private hidden chain-of-thought. It stores structured decisions, tool calls, results, state mutations, verifier failures, and accepted artifacts. This is sufficient for audit and reverse Skill mining.
