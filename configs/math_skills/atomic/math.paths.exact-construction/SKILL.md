---
name: math.paths.exact-construction
---
# Lattice paths with forbidden states

## Construction
Count lattice paths from (0,0) to (width,height), using only unit right/up moves and avoiding listed lattice POINTS. Specify coordinates and distinguish forbidden points from cells.

## Exact verification contract
Use `skill_python` with skill_id `math.paths.exact-construction`, script `verify.py`, and input
`{"family":"paths","parameters":...}`. The parameters schema is illustrated below.
Choose NEW parameters for each task; this fixture is not a question to copy.

```json
{
  "family": "paths",
  "parameters": {
    "width": 2,
    "height": 2,
    "forbidden": [
      [
        1,
        1
      ]
    ]
  }
}
```

The verifier computes an exact scalar answer, or fails on unsupported/ill-defined inputs.
Read tools/verify.py if necessary for bounds. Preserve the exact input in
TaskBundle.metadata.math_spec. Every parameter, domain and target must agree
with the question. A matching numeric output alone does not validate prose.

## Compose
Combine this construction with domain checking and failure-driven revision skills.
You may rephrase conditions but must not add constraints absent from the witness.

## Workflow
Load skills; choose parameters; save the draft with working_state_edit(stage="draft");
execute the verifier; inspect feasibility/uniqueness/boundaries; update stage="check";
revise genuinely detected defects and recompute after changes; run self_check_artifact;
submit the exact candidate only after passing. Include concise tool-grounded rationale.

## Limitations
This is a bounded pilot construction, not an official paper skill or a guarantee
that the generated problem is hard. The fixture report covers numerical behavior;
the blind API verifier independently checks the final natural-language problem.
