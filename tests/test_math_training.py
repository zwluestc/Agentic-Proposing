from itertools import product, permutations, combinations
from fractions import Fraction
import math
import subprocess
import sys
import json
from pathlib import Path

import pytest

from agentic_proposing.artifacts.math_gate import MathConstructionGate
from agentic_proposing.core.models import TaskBundle
from agentic_proposing.providers.base import parse_json_object
from agentic_proposing.training.math_parameters import suggest
from agentic_proposing.training.math_tasks import solve


def test_authored_fixtures_are_recomputed():
    library = Path(__file__).resolve().parents[1] / "configs/math_skills/atomic"
    fixtures = list(library.glob("*/tests/fixtures.jsonl"))
    assert len(fixtures) == 40
    for path in fixtures:
        for line in path.read_text().splitlines():
            fixture = json.loads(line)
            assert solve(fixture["input"])["answer"] == fixture["expected_answer"]


def test_digit_dp_agrees_with_exhaustive_enumeration():
    for leading_zero in (True,False):
        for residue in range(5):
            p = {"length":4,"alphabet":3,"modulus":5,"residue":residue,"leading_zero":leading_zero}
            expected = sum(1 for digits in product(range(3),repeat=4)
                           if (leading_zero or digits[0] != 0)
                           and all(a != b for a,b in zip(digits,digits[1:]))
                           and sum(digits) % 5 == residue)
            assert solve({"family":"digit_count","parameters":p})["answer"] == str(expected)


def test_invalid_constructions_fail_closed():
    with pytest.raises(ValueError,match="unique"):
        solve({"family":"linear","parameters":{"a":1,"b":2,"c":2,"d":4,"r":3,"s":6,"u":1,"v":1}})
    with pytest.raises(ValueError,match="unreachable"):
        solve({"family":"graph","parameters":{"vertices":3,"edges":[[0,1,3]],"directed":True,"source":0,"target":2}})


def test_gate_rejects_wrong_claimed_answer():
    spec = {"family":"paths","parameters":{"width":2,"height":2,"forbidden":[[1,1]]}}
    bundle = TaskBundle("math","calculation","Count lattice paths avoiding the center.","6",metadata={"math_spec":spec})
    assert not MathConstructionGate().check(bundle,None).passed
    bundle.answer = "2"
    assert MathConstructionGate().check(bundle,None).passed


def test_verifier_parser_ignores_earlier_json_fragments():
    verdict = {"valid":True,"answer":"2","ambiguities":[],"fatal_errors":[],"justification":"Checked."}
    text = 'Candidates [] and intermediate {"x":1}. Final:\n```json\n'+json.dumps(verdict)+'\n```'
    assert parse_json_object(text,set(verdict)) == verdict
    with pytest.raises(ValueError):
        parse_json_object('[] {"valid":true',set(verdict))


def test_parameter_suggestions_are_feasible_and_varied():
    families = [path.parent.parent.name.split(".")[1] for path in
                (Path(__file__).resolve().parents[1]/"configs/math_skills/atomic").glob("*/tests/fixtures.jsonl")]
    assert len(families) == 40
    for family in families:
        specs = [suggest(family,i) for i in range(30)]
        assert len({json.dumps(s,sort_keys=True) for s in specs}) == len(specs)
        for spec in specs:
            assert solve(spec)["answer"]


def test_permutation_constructions_against_enumeration():
    for n in range(2,7):
        objects = list(permutations(range(n)))
        deranged = sum(all(value != i for i,value in enumerate(obj)) for obj in objects)
        involutions = sum(all(obj[obj[i]] == i for i in range(n)) for obj in objects)
        assert solve({"family":"derangements","parameters":{"n":n}})["answer"] == str(deranged)
        assert solve({"family":"involutions","parameters":{"n":n}})["answer"] == str(involutions)
        for k in range(n*(n-1)//2+1):
            expected = sum(sum(a>b for i,a in enumerate(obj) for b in obj[i+1:]) == k for obj in objects)
            assert solve({"family":"permutation_inversions","parameters":{"n":n,"inversions":k}})["answer"] == str(expected)


def test_counting_constructions_against_enumeration():
    for n in range(2,7):
        for m in range(1,min(n,3)+1):
            count = sum(len(set(obj)) == m for obj in product(range(m),repeat=n))
            assert solve({"family":"surjections","parameters":{"domain_size":n,"codomain_size":m}})["answer"] == str(count)
        necklaces = {min(tuple(obj[i:]+obj[:i]) for i in range(n)) for obj in product(range(2),repeat=n)}
        assert solve({"family":"necklaces","parameters":{"length":n,"colors":2}})["answer"] == str(len(necklaces))
    for a in range(3,6):
        for b in range(1,a):
            count = 0
            for positions in combinations(range(a+b),b):
                selected = set(positions)
                prefix = 0
                valid = True
                for i in range(a+b):
                    prefix += -1 if i in selected else 1
                    valid &= prefix > 0
                count += valid
            assert solve({"family":"ballot","parameters":{"a_votes":a,"b_votes":b}})["answer"] == str(count)


def test_number_theory_and_probability_against_direct_computation():
    for n in range(2,60):
        assert solve({"family":"totient","parameters":{"n":n}})["answer"] == str(sum(math.gcd(n,k)==1 for k in range(1,n+1)))
        for k in range(1,n):
            value = math.comb(n,k)
            zeros = len(str(value))-len(str(value).rstrip("0"))
            assert solve({"family":"trailing_zeros_binomial","parameters":{"n":n,"k":k}})["answer"] == str(zeros)
    for dice in range(1,4):
        expected = Fraction(sum(max(obj) for obj in product(range(1,5),repeat=dice)),4**dice)
        assert solve({"family":"expected_max","parameters":{"dice":dice,"sides":4}})["answer"] == str(expected)


def test_geometry_fails_on_invalid_inputs_and_respects_orientation():
    with pytest.raises(ValueError,match="simple"):
        solve({"family":"lattice_polygon_interior","parameters":{"vertices":[[0,0],[3,3],[0,3],[3,0]]}})
    with pytest.raises(ValueError,match="Degenerate"):
        solve({"family":"circumradius_squared","parameters":{"points":[[0,0],[1,1],[2,2]]}})
    rectangle = [[0,0],[5,0],[5,4],[0,4]]
    for points in (rectangle,list(reversed(rectangle))):
        assert solve({"family":"lattice_polygon_interior","parameters":{"vertices":points}})["answer"] == "12"
    with pytest.raises(ValueError,match="sum to one"):
        solve({"family":"markov_chain","parameters":{"transition":[["1/2","1/3"],["0","1"]],"start":0,"target":1,"steps":2}})


def test_polynomial_and_probability_cross_checks():
    # x^k mod (x-1)(x-2): the linear interpolant is determined by
    # its values 1 and 2^k, independently of quotient-ring iteration.
    for exponent in range(2,20):
        for coefficient,expected in [(0,2-2**exponent),(1,2**exponent-1)]:
            assert solve({"family":"polynomial_remainder","parameters":{"b":-3,"c":2,"exponent":exponent,"coefficient":coefficient}})["answer"] == str(expected)
    matrix = [["1/2","1/3","1/6"],["1/4","1/2","1/4"],["1/3","1/3","1/3"]]
    for steps in (1,2,5,10):
        probabilities = [Fraction(solve({"family":"markov_chain","parameters":{"transition":matrix,"start":0,"target":t,"steps":steps}})["answer"]) for t in range(3)]
        assert sum(probabilities) == 1
    # This triangle lies on the circle centered at (2,1), radius sqrt(5).
    assert solve({"family":"circumradius_squared","parameters":{"points":[[0,0],[4,0],[1,3]]}})["answer"] == "5"


def test_every_skill_executable_is_bound_to_its_family():
    library = Path(__file__).resolve().parents[1]/"configs/math_skills/atomic"
    for executable in library.glob("*/tools/verify.py"):
        family = executable.parent.parent.name.split(".")[1]
        other = "linear" if family != "linear" else "quadratic"
        completed = subprocess.run([sys.executable,"-I",str(executable)],input=json.dumps(suggest(other,1)),text=True,capture_output=True)
        assert completed.returncode != 0
        assert "does not match" in json.loads(completed.stdout)["error"]


def test_large_library_metadata_stays_complete_and_prioritizes_active_skill(tmp_path):
    from agentic_proposing.training.runner import _prepare_workspace
    from agentic_proposing.core.store import WorkspaceLayout,RunStore
    from agentic_proposing.core.models import RunState
    from agentic_proposing.core.context import ContextManager
    from agentic_proposing.skills.registry import SkillRegistry
    from agentic_proposing.tools.base import ToolRegistry
    library = Path(__file__).resolve().parents[1]/"configs/math_skills/atomic"
    workspace = _prepare_workspace({"workspace":str(tmp_path),"skill_library":str(library)})
    layout = WorkspaceLayout(workspace)
    registry = SkillRegistry(layout)
    state = RunState.create("Construct a factorial valuation problem",workspace)
    active = "math.valuation_factorial.exact-construction"
    state.active_skill_ids = [active]
    messages,_ = ContextManager(ToolRegistry(),registry).materialize(state,RunStore(layout,state.run_id))
    metadata = messages[0]["content"].split("# Discoverable Skill Metadata\n",1)[1]
    parsed = json.loads(metadata)
    assert len(parsed) == 50
    assert parsed[0]["skill_id"] == active
    assert len(metadata) <= 18000
