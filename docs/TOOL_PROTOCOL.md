# Tool protocol

Every tool has:

- name and discovery description;
- JSON input/output schemas;
- side-effect class;
- read-only and parallelization annotations;
- deferred/eager discovery flag;
- tags;
- permission and hook interception;
- bounded result returned to context;
- optional full artifact written to the run workspace.

## Result envelope

```json
{
  "call_id": "call_...",
  "tool_name": "evidence_search",
  "content": "short model-facing summary",
  "data": {},
  "is_error": false,
  "terminal": false,
  "state_updates": {},
  "artifact_refs": []
}
```

Failures are observations, not Python crashes from the harness. The model receives `is_error=true` and can retry, change tools, prune/rebind a Skill, or refine the artifact.

## Included capabilities

- `tool_search`: discover and enable deferred tool schemas.
- `working_state_edit`, `artifact_write`: persist compact run state or a managed full artifact.
- `workspace_glob`, `workspace_read`, `workspace_grep`: bounded inspection within configured roots.
- `skill_search`, `skill_load`, `skill_compose`, `skill_stage`, `skill_lineage`.
- `skill_prune`, `skill_promote`, `skill_deprecate`, `skill_version`.
- `benchmark_parse`, `corpus_ingest`, `evidence_search`, `evidence_fetch`.
- `web_search`, `web_fetch`, `evidence_ingest_url`: configured search, safe bounded public fetch, and provenance-preserving EvidenceStore acquisition.
- `python_eval`: pure AST-allowlisted expressions for isolated calculations.
- `python_session_create`, `python_session_exec`, `python_session_close`: stateful restricted analysis within one run.
- `skill_python`: trusted versioned Skill script execution.
- `tool_catalog_refresh`: transactional refresh of configured JSON-over-HTTP external providers.
- `self_check_artifact`: nonterminal deterministic gates and exact-candidate fingerprint.
- `verifier_committee`: optional isolated privileged votes plus blind solve and second audit.
- `submit_artifact`: the only gated terminal action; optionally requires a fresh self-check.

## Safety boundary

Side-effect permissions are checked before read-only permissions: a read-only network or process tool still requires its dedicated capability flag.

`python_eval` is not a general interpreter. Persistent Python sessions add state, statements, selected imports, bounded output, per-run directories, and process cleanup; `skill_python` remains for trusted versioned repository scripts. Neither mechanism is claimed to be a hostile-code OS sandbox. Untrusted arbitrary code requires a container or equivalent isolation supplied outside this repository.

Public `web_fetch` and `evidence_ingest_url` reject private/local/reserved network targets and recheck redirect destinations. External provider endpoints are operator-configured trusted integrations and still pass through the network permission layer. The provider abstraction is transport-extensible, but the current implementation is JSON-over-HTTP rather than a claim of complete MCP protocol compatibility.

