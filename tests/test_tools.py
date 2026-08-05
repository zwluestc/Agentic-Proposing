from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.config import RuntimeConfig
from agentic_proposing.core.models import RunState, ToolCall
from agentic_proposing.core.store import RunStore
from agentic_proposing.providers.scripted import ScriptedTeacher
from agentic_proposing.runtime import build_runtime
from agentic_proposing.tools.base import ToolContext


class ToolTests(unittest.TestCase):
    def test_tool_search_enables_deferred_schema(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            runtime = build_runtime(temporary, ScriptedTeacher(), RuntimeConfig(skill_mining={"enabled": False}))
            state = RunState.create("discover", temporary)
            run_store = RunStore(runtime.layout, state.run_id)
            context = ToolContext(runtime.layout, run_store, state, runtime.harness.services)
            result = runtime.dispatcher.execute(
                [ToolCall("tool_search", {"query": "evidence search", "load": ["evidence_search"]})], context
            )[0]
            self.assertFalse(result.is_error)
            self.assertIn("evidence_search", state.enabled_tool_names)
            _, tools = runtime.context_manager.materialize(state, run_store)
            names = {tool["function"]["name"] for tool in tools}
            self.assertIn("evidence_search", names)

    def test_python_eval_rejects_imports(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            runtime = build_runtime(temporary, ScriptedTeacher(), RuntimeConfig(skill_mining={"enabled": False}))
            state = RunState.create("calculate", temporary)
            context = ToolContext(runtime.layout, RunStore(runtime.layout, state.run_id), state, runtime.harness.services)
            good = runtime.dispatcher.execute([ToolCall("python_eval", {"expression": "sqrt(16)+2"})], context)[0]
            self.assertEqual(good.data["result"], 6.0)
            bad = runtime.dispatcher.execute([ToolCall("python_eval", {"expression": "__import__('os')"})], context)[0]
            self.assertTrue(bad.is_error)


if __name__ == "__main__":
    unittest.main()

