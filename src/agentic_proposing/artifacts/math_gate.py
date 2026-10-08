"""Opt-in exact grading for the declared math construction witness."""
from .verification import CheckResult
from agentic_proposing.training.math_tasks import solve
from agentic_proposing.training.prober import answers_match


class MathConstructionGate:
    name = "math_construction"

    def check(self, bundle, context):
        spec = bundle.metadata.get("math_spec")
        if not isinstance(spec, dict):
            return CheckResult(self.name, False, "fatal", "metadata.math_spec is required")
        expected = solve(spec)
        if context is not None:
            skill_id = f"math.{spec['family']}.exact-construction"
            if skill_id not in bundle.skill_lineage or skill_id not in context.state.loaded_skill_ids:
                return CheckResult(self.name, False, "fatal", f"Load {skill_id} and include it in skill_lineage")
            executed = any(
                event.get("event_type") == "tool_succeeded"
                and event["payload"].get("tool_name") == "skill_python"
                and event["payload"].get("data",{}).get("result",{}).get("family") == spec["family"]
                and event["payload"].get("data",{}).get("result",{}).get("verified_parameters") == spec["parameters"]
                for event in context.run_store.events.read()
            )
            if not executed:
                return CheckResult(self.name, False, "fatal", "Run skill_python on the exact final math_spec before self-check/submission")
        passed = answers_match(bundle.answer, expected["answer"])
        return CheckResult(self.name, passed, "fatal", "Declared construction answer recomputed" if passed else "Answer disagrees with exact construction", expected)
