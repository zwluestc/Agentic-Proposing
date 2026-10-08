#!/usr/bin/env python3
"""Build reproducible, hand-authored math Skills and synthesis objectives."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agentic_proposing.training.math_tasks import solve
from math_skill_catalog import EXTRA_FAMILIES, SUPPORT_SKILLS

FAMILIES = {
    "linear": ("Linear constraints and elimination", "Construct two independent linear equations and ask for a weighted combination of their unique solution. Translate an explicit story into both equations; avoid merely asking to solve one equation.", {"a":2,"b":1,"c":1,"d":-1,"r":11,"s":1,"u":1,"v":1}, "7", {"a":2,"b":1,"c":1,"d":-1,"r":13,"s":2,"u":1,"v":1}, "8"),
    "quadratic": ("Integer roots and symmetric transformations", "State a quadratic and an inclusive integer domain, then ask for the sum of the stated powers of ALL DISTINCT integer roots. Specify distinctness and the domain.", {"a":1,"b":-5,"c":6,"lo":0,"hi":10,"power":2}, "13", {"a":1,"b":1,"c":-6,"lo":-10,"hi":10,"power":3}, "-19"),
    "crt": ("Bounded modular constraints", "Combine two to four remainder conditions on an integer in an explicit inclusive interval. Ask for its minimum, the count, or the sum of all matching integers; the target must match parameters.target.", {"moduli":[3,5],"residues":[1,2],"lo":1,"hi":30,"target":"minimum"}, "7", {"moduli":[3,5],"residues":[1,2],"lo":1,"hi":30,"target":"count"}, "2"),
    "digit_count": ("Digit-string constraint composition", "Count ordered digit strings of fixed length, alphabet 0 through alphabet-1, unequal adjacent digits, and a digit-sum residue. Explicitly state whether leading zero is allowed. Explain a DP rather than listing arbitrary strings.", {"length":3,"alphabet":2,"modulus":3,"residue":2,"leading_zero":False}, "1", {"length":2,"alphabet":3,"modulus":2,"residue":1,"leading_zero":True}, "4"),
    "paths": ("Lattice paths with forbidden states", "Count lattice paths from (0,0) to (width,height), using only unit right/up moves and avoiding listed lattice POINTS. Specify coordinates and distinguish forbidden points from cells.", {"width":2,"height":2,"forbidden":[[1,1]]}, "2", {"width":2,"height":2,"forbidden":[]}, "6"),
    "recurrence": ("Affine recurrences and modular reduction", "Define f_0, f_1 and f_n=a*f_(n-1)+b*f_(n-2)+c for n>=2, and ask for f_n modulo a positive modulus. State indexing and every coefficient explicitly.", {"n":10,"a":1,"b":1,"c":0,"f0":0,"f1":1,"modulus":1000}, "55", {"n":3,"a":2,"b":0,"c":1,"f0":0,"f1":1,"modulus":100}, "7"),
    "probability": ("Conditional finite probability", "Use independent uniform dice with sides labels 1..sides. Ask P(sum=target_sum | sum>=threshold) for dice dice. State independence, uniformity and the conditioning event; return an exact reduced fraction.", {"dice":2,"sides":6,"threshold":10,"target_sum":12}, "1/6", {"dice":2,"sides":2,"threshold":3,"target_sum":3}, "2/3"),
    "optimization": ("Finite integer optimization", "Maximize u*x+v*y for integer 0<=x,y<=bound under a*x+b*y<=budget and c*x+d*y<=budget. Ask only for the optimal VALUE, which remains unique even if maximizers tie. Make integrality and both constraints explicit.", {"bound":5,"a":1,"b":1,"c":1,"d":2,"budget":5,"u":2,"v":3}, "10", {"bound":5,"a":1,"b":1,"c":1,"d":2,"budget":5,"u":1,"v":3}, "7"),
    "graph": ("Weighted graph shortest paths", "Specify a finite graph with vertices 0..vertices-1 and the COMPLETE weighted edge list. State directedness and ask for the shortest-distance VALUE from source to target. Multiple optimal paths do not invalidate a unique distance.", {"vertices":3,"edges":[[0,1,2],[1,2,3],[0,2,9]],"directed":False,"source":0,"target":2}, "5", {"vertices":3,"edges":[[0,1,2],[1,2,3],[0,2,9]],"directed":True,"source":2,"target":2}, "0"),
    "rational_sum": ("Exact telescoping rational sums", "Ask for scale times sum_{k=start}^end 1/[k*(k+shift)]. Use an inclusive range, exact rational arithmetic, and a nontrivial shift or range. Explain the cancellation structure.", {"start":1,"end":3,"shift":1,"scale":1}, "3/4", {"start":2,"end":4,"shift":1,"scale":2}, "3/5"),
}

VARIANTS = [
    "Use a direct mathematical formulation with at least two interacting constraints.",
    "Use a concise self-contained real-world story; explicitly translate all story conditions into the exact mathematical construction.",
    "Choose a boundary case that tests domain/index/endpoint reasoning without creating ambiguity.",
    "Choose parameters requiring a nontrivial intermediate computation; avoid a one-operation exercise.",
    "Draft a candidate, inspect the most likely ambiguity or computational error, and revise only if a real defect is found. Do not invent a failed check.",
    "Increase structural difficulty within the supported construction, then check feasibility and correctness before finalizing.",
]

FAMILIES.update(EXTRA_FAMILIES)


def main():
    library = ROOT / "configs/math_skills/atomic"
    script = (ROOT / "src/agentic_proposing/training/math_tasks.py").read_text()
    report, seeds = [], []
    for family, (name, method, p1, expected1, p2, expected2) in FAMILIES.items():
        skill_id = f"math.{family}.exact-construction"
        path = library / skill_id
        (path / "tools").mkdir(parents=True, exist_ok=True)
        (path / "tests").mkdir(exist_ok=True)
        (path / "verification").mkdir(exist_ok=True)
        # Bind each executable to its own family; arbitrary cross-family inputs
        # are rejected even though the implementation source is shared.
        bound_script = script.replace(
            'print(json.dumps(solve(json.load(sys.stdin)),ensure_ascii=False))',
            'spec = json.load(sys.stdin)\n'
            f'        if spec.get("family") != {family!r}:\n'
            '            raise ValueError("Input family does not match this Skill")\n'
            '        print(json.dumps(solve(spec),ensure_ascii=False))',
        )
        (path / "tools/verify.py").write_text(bound_script)
        fixtures = [{"input":{"family":family,"parameters":p},"expected_answer":e} for p,e in [(p1,expected1),(p2,expected2)]]
        for fixture in fixtures:
            assert solve(fixture["input"])["answer"] == fixture["expected_answer"]
            result = subprocess.run([sys.executable,"-I",str(path / "tools/verify.py")], input=json.dumps(fixture["input"]), text=True, capture_output=True, check=True)
            assert json.loads(result.stdout)["answer"] == fixture["expected_answer"]
        validation = {"passed":True,"validation_type":"exact_constructor_fixture_replay","fixtures_passed":len(fixtures),"scope":"Two independent hand-calculated parameter fixtures; not a proof of arbitrary natural-language validity or hardness."}
        manifest = {"skill_id":skill_id,"name":name,"version":"2.0.0","kind":"atomic","status":"verified","description":method,"intent":name,"triggers":[family],"inputs":["metadata.math_spec.parameters"],"outputs":["scalar exact answer"],"difficulty_effect":{"mechanism":"constraint interaction; parameter dependent, not measured"},"tools":["skill_python","python_eval"],"evidence_contract":{"knowledge_dependency":False},"source_refs":[{"type":"hand_authored","source":"scripts/prepare_math_training.py"}],"metrics":validation,"tags":["math",family,"exact"]}
        (path / "manifest.yaml").write_text(json.dumps(manifest,indent=2))
        (path / "tests/fixtures.jsonl").write_text("".join(json.dumps(f)+"\n" for f in fixtures))
        (path / "verification/validation.json").write_text(json.dumps(validation,indent=2))
        (path / "lineage.json").write_text(json.dumps({"derivation":"hand_authored_math_constructor","official_paper_skill":False}))
        body = f"""---
name: {skill_id}
---
# {name}

## Construction
{method}

## Exact verification contract
Use `skill_python` with skill_id `{skill_id}`, script `verify.py`, and input
`{{"family":"{family}","parameters":...}}`. The parameters schema is illustrated below.
Choose NEW parameters for each task; this fixture is not a question to copy.

```json
{json.dumps(fixtures[0]['input'],indent=2)}
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
"""
        (path / "SKILL.md").write_text(body)
        report.append({"skill_id":skill_id,**validation})
        for index, variant in enumerate(VARIANTS):
            support = SUPPORT_SKILLS[2+(list(FAMILIES).index(family)+index) % (len(SUPPORT_SKILLS)-2)][0]
            seeds.append({"id":f"{family}-{index+1}","category":family,"skill_ids":[skill_id,"math.validation.domain-check","math.revision.failure-driven","math."+support],"objective":f"Construct one original English-language calculation problem with a unique scalar answer using {name}. {method} {variant} Use new parameters, never copy the fixture. Set metadata.knowledge_dependency=false and metadata.math_spec to the EXACT skill_python input. Include all domains and constraints in the question. Use task_type=calculation. Load relevant skills, record stage transitions, execute skill_python, inspect its result, self-check, then submit. Keep observable decisions and rationale concise; do not write hidden chain-of-thought."})
    for suffix,name,body in SUPPORT_SKILLS:
        skill_id = "math."+suffix
        path = library / skill_id
        path.mkdir(parents=True,exist_ok=True)
        manifest = {"skill_id":skill_id,"name":name,"version":"2.0.0","kind":"atomic","status":"curated","description":body,"intent":name,"tools":["skill_python","python_eval"],"source_refs":[{"type":"hand_authored"}],"metrics":{"scope":"Process guidance; not independently replay-certified"},"tags":["math","process"]}
        (path / "manifest.yaml").write_text(json.dumps(manifest,indent=2))
        (path / "SKILL.md").write_text(f"---\nname: {skill_id}\n---\n# {name}\n\n{body}\n")
    destination = ROOT / "configs/proposer_seeds.math.jsonl"
    destination.write_text("".join(json.dumps(s)+"\n" for s in seeds))
    (ROOT / "configs/math_skills/validation_report.json").write_text(json.dumps({"skills":report,"supporting_skills":len(SUPPORT_SKILLS),"seeds":len(seeds),"official_library":False,"library_version":"2.0.0"},indent=2))
    catalogue = "# Math Skill catalogue v2\n\n40 bounded numerical constructions and 10 curated process skills.\n\n| Family | Construction | Skill ID |\n|---|---|---|\n"
    catalogue += "".join(f"| {family} | {values[0]} | math.{family}.exact-construction |\n" for family,values in FAMILIES.items())
    catalogue += "\n## Process guidance\n\n"+"".join(f"- `math.{suffix}`: {name}\n" for suffix,name,_ in SUPPORT_SKILLS)
    (ROOT / "configs/math_skills/README.md").write_text(catalogue)
    print(json.dumps({"constructor_skills":len(report),"supporting_skills":len(SUPPORT_SKILLS),"fixture_replays":2*len(report),"seeds":len(seeds)}))


if __name__ == "__main__":
    main()
