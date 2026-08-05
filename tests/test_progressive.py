from __future__ import annotations

from tempfile import TemporaryDirectory
import unittest

from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.config import RuntimeConfig
from agentic_proposing.core.models import RunState, ToolCall
from agentic_proposing.core.store import RunStore
from agentic_proposing.providers.scripted import ScriptedTeacher
from agentic_proposing.runtime import build_runtime
from agentic_proposing.tools.base import ToolContext


class ProgressiveDisclosureTests(unittest.TestCase):
    def test_loading_skill_enables_declared_tool_schema(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            runtime = build_runtime(
                temporary, ScriptedTeacher(), RuntimeConfig(skill_mining={"enabled": False})
            )
            state = RunState.create("load a skill", temporary)
            store = RunStore(runtime.layout, state.run_id)
            context = ToolContext(runtime.layout, store, state, runtime.harness.services)
            result = runtime.dispatcher.execute(
                [ToolCall("skill_load", {"skill_ids": ["med.calc.two-stage-clearance"]})], context
            )[0]
            self.assertFalse(result.is_error)
            self.assertIn("python_eval", state.enabled_tool_names)
            messages, schemas = runtime.context_manager.materialize(state, store)
            self.assertIn("Skill of Two-stage Clearance", messages[0]["content"])
            names = {item["function"]["name"] for item in schemas}
            self.assertIn("python_eval", names)


if __name__ == "__main__":
    unittest.main()

