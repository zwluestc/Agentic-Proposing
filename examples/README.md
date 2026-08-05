# Examples

1. Initialize and run the offline harness:

   `agentic-proposing demo --workspace ./demo-workspace`

2. Compile a recursive composite:

   `agentic-proposing compose examples/composite.two-stage-choice.json --workspace ./demo-workspace`

3. Ingest the synthetic textbook fragment:

   `agentic-proposing ingest examples/sources/synthetic_renal_notes.md --workspace ./demo-workspace`

4. Configure a real SOTA Teacher by copying one of the config files, selecting a current model, and setting the corresponding API key environment variable. The examples require a fresh `self_check_artifact` before terminal submit.

   For live references, choose `brave`, `tavily`, or `searxng` under `web`, enable the narrow network permission, and set an authoritative domain allowlist. See `docs/CONFIGURATION.md`.

   For multi-cell analysis, enable process permission and let the agent discover `python_session_create`, `python_session_exec`, and `python_session_close` through `tool_search`.

5. Run recursive composition with independent validation replays:

   `agentic-proposing evolve "compose evidence-grounded medical reasoning skills" --workspace ./demo-workspace --rounds 3 --replays-per-composition 2`

6. Inspect and manage the registry:

   `agentic-proposing skills list --workspace ./demo-workspace`

   `agentic-proposing skills version med.calc.two-stage-clearance 1.1.0 --change-summary "tighten unit checks" --workspace ./demo-workspace`

   After replay validation, promote the staged version with `--replace-existing`; the prior version is archived.

The source examples are synthetic and are not medical guidance.

