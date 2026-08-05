# Testing and acceptance

The offline test suite does not call a remote model. `ScriptedTeacher` emits deterministic native tool calls and JSON responses.

Required acceptance coverage:

- event loop continues after plain text and after rejected submit;
- accepted `submit_artifact` is the only successful terminal path;
- run state, events, checkpoints, artifacts, and final TaskBundle persist;
- Skill metadata indexing does not read bodies;
- selected bodies load on demand;
- same Skill type may appear twice with different instance IDs/bindings;
- duplicate instance IDs and composition cycles are rejected;
- textbook/benchmark parsers preserve source/row/chunk provenance;
- evidence retrieval returns exact IDs and locators;
- knowledge-dependent tasks without resolvable evidence are rejected;
- generated candidates are staged, not silently promoted;
- post-acceptance mining records task/run lineage;
- duplicate detection and replay reports gate promotion;
- a promoted C becomes a real parent of D in the next recursive round;
- blind verification matches the privileged majority, not a minority answer;
- second-audit vote serialization and resolution work;
- Skill version replacement archives the previous active package and preserves lineage;
- reverse Skill-mining failure cannot invalidate an already accepted Task;
- read-only network/process tools cannot bypass policy, `ask` uses an explicit resolver, and deny wins;
- workspace Read/Glob/Grep enforce roots and oversized results spill to managed artifacts;
- persistent Python sessions preserve state, reject forbidden imports, and close their pipes/processes;
- Brave/Tavily/SearXNG adapters normalize results and public web fetch blocks private targets;
- Web-to-Evidence ingestion preserves URL/time/hash provenance, including duplicate acquisition;
- nonterminal self-check passes only the exact candidate fingerprint required by terminal submit;
- external provider refresh is validated before the dynamic registry is changed.

Run:

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

