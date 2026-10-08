---
name: math.recurrence.exact-construction
---
# Affine recurrences and modular reduction

## Construction
Define f_0, f_1 and f_n=a*f_(n-1)+b*f_(n-2)+c for n>=2, and ask for f_n modulo a positive modulus. State indexing and every coefficient explicitly.

## Exact verification contract
Use `skill_python` with skill_id `math.recurrence.exact-construction`, script `verify.py`, and input
`{"family":"recurrence","parameters":...}`. The parameters schema is illustrated below.
Choose NEW parameters for each task; this fixture is not a question to copy.

```json
{
  "family": "recurrence",
  "parameters": {
    "n": 10,
    "a": 1,
    "b": 1,
    "c": 0,
    "f0": 0,
    "f1": 1,
    "modulus": 1000
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
