# Skill package format

```text
<skill-id>/
  manifest.yaml
  SKILL.md
  lineage.json
  metrics.json                 # present after validation/promotion
  composition.yaml             # composite only
  references/
  tools/
  verification/
  tests/
    positive.jsonl
    negative.jsonl
    transfer.jsonl
```

## Discovery layer: `manifest.yaml`

The manifest is JSON-compatible YAML so the base installation can read it without a YAML dependency. It contains identity/version/status, a short trigger description, intent, typed inputs and outputs, difficulty effects, tool names, evidence contract, composition summary, source references, metrics, and tags.

Registry indexing reads only manifests. This is the progressive-disclosure boundary.

## Procedure layer: `SKILL.md`

The body contains:

1. Structure/formal backbone and preconditions.
2. Action/executable construction steps.
3. Effect/target outcome.
4. Typed bindings and outputs.
5. Difficulty controls.
6. Evidence and tool-use protocol.
7. Failure detection, refinement, and rejection rules.

It should not contain replaceable medical facts or copied benchmark instances.

## Tool layer

Skill tools are versioned scripts or declarative contracts. `skill_python` runs only an existing `.py` file under the selected Skill's `tools/` directory. Arbitrary model-supplied scripts are rejected by that tool.

## Lineage

Atomic extraction records source type, document/item locator, reasoning nodes, abstractions, extractor/model version, and tests. Composite lineage records every parent occurrence, binding, edge, and novelty claim. Task-mined Skills additionally record accepted task ID and run ID.

## Candidate lifecycle

```text
candidate -> staged -> replayed -> verified -> promoted
                         |             |
                         +-> rejected  +-> deprecated/versioned
```

Staging is intentionally separate from the searchable verified registry. Promotion requires a structured validation report with `passed=true`; near-duplicate candidates are blocked unless explicitly audited. A new version is forked into staging with `versioned_from` lineage. Replacement promotion archives the active package under `skills/deprecated/<id>@<old-version>` and rolls back the archive move if activation fails.

