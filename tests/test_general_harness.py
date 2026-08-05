from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

from agentic_proposing.bootstrap import bootstrap_demo_workspace
from agentic_proposing.config import RuntimeConfig
from agentic_proposing.core.models import ModelTurn, RunState, ToolCall
from agentic_proposing.core.store import RunStore
from agentic_proposing.providers.scripted import ScriptedTeacher
from agentic_proposing.runtime import build_runtime
from agentic_proposing.tools.base import ToolContext
from agentic_proposing.tools.external_tools import ToolProviderManager, providers_from_config


def bundle(answer: int = 25) -> dict:
    return {
        "task_id": "checked-task",
        "task_type": "calculation",
        "question": "Using the supplied formula, calculate the unique scalar result for the numeric input four.",
        "answer": answer,
        "rationale": "One hundred divided by four equals twenty-five.",
        "rubric": [],
        "evidence": [],
        "skill_lineage": ["med.calc.two-stage-clearance"],
        "difficulty": {"estimated": 1},
        "metadata": {"knowledge_dependency": False},
    }


def context_for(runtime, root: str) -> ToolContext:
    state = RunState.create("test general harness", root)
    return ToolContext(
        runtime.layout,
        RunStore(runtime.layout, state.run_id),
        state,
        runtime.harness.services,
    )


class GeneralHarnessTests(unittest.TestCase):
    def test_workspace_tools_are_bounded_and_long_results_spill(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            source = Path(temporary) / "sources" / "large.txt"
            source.write_text("needle " * 1000, encoding="utf-8")
            runtime = build_runtime(
                temporary,
                ScriptedTeacher(),
                RuntimeConfig(tools={"max_inline_chars": 300}, skill_mining={"enabled": False}),
            )
            context = context_for(runtime, temporary)
            read = runtime.dispatcher.execute(
                [ToolCall("workspace_read", {"path": "sources/large.txt", "max_chars": 20000})], context
            )[0]
            self.assertFalse(read.is_error)
            self.assertTrue(read.data["truncated"])
            self.assertTrue(read.artifact_refs)
            grep = runtime.dispatcher.execute(
                [ToolCall("workspace_grep", {"query": "needle", "include": "sources/*.txt"})], context
            )[0]
            self.assertFalse(grep.is_error)
            outside = runtime.dispatcher.execute(
                [ToolCall("workspace_read", {"path": str(Path(temporary).parent)})], context
            )[0]
            self.assertTrue(outside.is_error)

    def test_persistent_python_session_keeps_state_and_closes(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            runtime = build_runtime(
                temporary,
                ScriptedTeacher(),
                RuntimeConfig(
                    permissions={"allow_read_only": True, "allow_process": True},
                    skill_mining={"enabled": False},
                ),
            )
            context = context_for(runtime, temporary)
            created = runtime.dispatcher.execute(
                [ToolCall("python_session_create", {"session_id": "analysis"})], context
            )[0]
            self.assertFalse(created.is_error)
            runtime.dispatcher.execute(
                [ToolCall("python_session_exec", {"session_id": "analysis", "code": "x = 2"})], context
            )
            calculated = runtime.dispatcher.execute(
                [ToolCall("python_session_exec", {"session_id": "analysis", "code": "x + 3"})], context
            )[0]
            self.assertEqual(calculated.data["result"], 5)
            forbidden = runtime.dispatcher.execute(
                [ToolCall("python_session_exec", {"session_id": "analysis", "code": "import os"})], context
            )[0]
            self.assertTrue(forbidden.is_error)
            runtime.dispatcher.execute(
                [ToolCall("python_session_close", {"session_id": "analysis"})], context
            )
            self.assertEqual(runtime.python_session_manager.list_for_run(context.state.run_id), [])

    def test_context_compaction_enforces_budget_without_orphan_tool_messages(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            runtime = build_runtime(
                temporary,
                ScriptedTeacher(),
                RuntimeConfig(
                    context={"max_context_chars": 64000, "max_transcript_chars": 4000},
                    skill_mining={"enabled": False},
                ),
            )
            context = context_for(runtime, temporary)
            for index in range(5):
                context.state.transcript.extend(
                    [
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [{"id": f"call-{index}", "type": "function"}],
                        },
                        {
                            "role": "tool",
                            "tool_call_id": f"call-{index}",
                            "name": "large_tool",
                            "content": "x" * 5000,
                        },
                    ]
                )
            runtime.context_manager.materialize(context.state, context.run_store)
            self.assertLessEqual(len(json.dumps(context.state.transcript)), 4000)
            if context.state.transcript:
                self.assertNotEqual(context.state.transcript[0]["role"], "tool")
            self.assertTrue(context.state.compact_summary)

    def test_harness_closes_unfinished_python_sessions_at_session_end(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            teacher = ScriptedTeacher(
                [
                    ModelTurn(tool_calls=[ToolCall("python_session_create", {"session_id": "temporary"})]),
                    ModelTurn(tool_calls=[ToolCall("submit_artifact", {"task_bundle": bundle()})]),
                ]
            )
            runtime = build_runtime(
                temporary,
                teacher,
                RuntimeConfig(
                    permissions={"allow_process": True},
                    harness={"max_turns": 4},
                    skill_mining={"enabled": False},
                ),
            )
            result = runtime.harness.run("Create and clean a temporary analysis session")
            self.assertEqual(result.status, "completed")
            self.assertEqual(runtime.python_session_manager.list_for_run(result.run_id), [])

    def test_self_check_is_nonterminal_and_submit_requires_fresh_candidate(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            runtime = build_runtime(
                temporary,
                ScriptedTeacher(),
                RuntimeConfig(
                    verification={"require_self_check": True},
                    skill_mining={"enabled": False},
                ),
            )
            context = context_for(runtime, temporary)
            premature = runtime.dispatcher.execute(
                [ToolCall("submit_artifact", {"task_bundle": bundle()})], context
            )[0]
            self.assertTrue(premature.is_error)
            checked = runtime.dispatcher.execute(
                [ToolCall("self_check_artifact", {"task_bundle": bundle()})], context
            )[0]
            self.assertFalse(checked.is_error)
            self.assertFalse(checked.terminal)
            stale = runtime.dispatcher.execute(
                [ToolCall("submit_artifact", {"task_bundle": bundle(24)})], context
            )[0]
            self.assertTrue(stale.is_error)
            accepted = runtime.dispatcher.execute(
                [ToolCall("submit_artifact", {"task_bundle": bundle()})], context
            )[0]
            self.assertTrue(accepted.terminal)
            self.assertFalse(accepted.is_error)

    def test_ask_permission_uses_resolver_but_deny_takes_precedence(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            approvals: list[dict] = []
            runtime = build_runtime(
                temporary,
                ScriptedTeacher(),
                RuntimeConfig(
                    permissions={"ask": ["workspace_glob"], "denied": ["workspace_read"]},
                    skill_mining={"enabled": False},
                ),
                permission_resolver=lambda request: approvals.append(request) is None,
            )
            context = context_for(runtime, temporary)
            approved = runtime.dispatcher.execute(
                [ToolCall("workspace_glob", {"pattern": "**/*.md"})], context
            )[0]
            self.assertFalse(approved.is_error)
            self.assertEqual(approvals[0]["tool_name"], "workspace_glob")
            denied = runtime.dispatcher.execute(
                [ToolCall("workspace_read", {"path": "sources/synthetic_renal_notes.md"})], context
            )[0]
            self.assertTrue(denied.is_error)
            self.assertEqual(len(approvals), 1)

    def test_web_page_can_be_ingested_with_provenance(self) -> None:
        with TemporaryDirectory() as temporary:
            bootstrap_demo_workspace(temporary)
            runtime = build_runtime(
                temporary,
                ScriptedTeacher(),
                RuntimeConfig(
                    permissions={"allow_network": True},
                    skill_mining={"enabled": False},
                ),
            )
            context = context_for(runtime, temporary)
            page = {
                "requested_url": "https://example.org/reference",
                "final_url": "https://example.org/reference",
                "content_type": "text/html; charset=utf-8",
                "title": "Reference",
                "text": "Clinical rule\n\nA unique synthetic rule supports the generated task.",
            }
            with patch("agentic_proposing.tools.web_evidence_tools.fetch_public_page", return_value=page):
                result = runtime.dispatcher.execute(
                    [ToolCall("evidence_ingest_url", {"url": page["requested_url"]})], context
                )[0]
            self.assertFalse(result.is_error)
            self.assertEqual(runtime.evidence_store.stats()["documents"], 1)
            self.assertIn("retrieved_at", result.data["document"])
            evidence_id = result.data["chunks"][0]["evidence_id"]
            self.assertEqual(runtime.evidence_store.fetch(evidence_id)["document_id"], result.data["document"]["document_id"])
            with patch("agentic_proposing.tools.web_evidence_tools.fetch_public_page", return_value=page):
                duplicate = runtime.dispatcher.execute(
                    [ToolCall("evidence_ingest_url", {"url": page["requested_url"]})], context
                )[0]
            self.assertTrue(duplicate.data["ingest"]["duplicate"])
            duplicate_id = duplicate.data["chunks"][0]["evidence_id"]
            self.assertEqual(duplicate_id, evidence_id)
            self.assertEqual(runtime.evidence_store.fetch(duplicate_id)["document_id"], result.data["document"]["document_id"])

    def test_static_external_provider_refreshes_dynamic_catalog(self) -> None:
        with TemporaryDirectory() as temporary:
            runtime = build_runtime(
                temporary,
                ScriptedTeacher(),
                RuntimeConfig(skill_mining={"enabled": False}),
            )
            providers = providers_from_config(
                {
                    "providers": [
                        {
                            "id": "local",
                            "tools": [
                                {
                                    "name": "domain_lookup",
                                    "description": "Configured lookup",
                                    "endpoint": "https://example.org/tool",
                                    "input_schema": {"type": "object"},
                                }
                            ],
                        }
                    ]
                }
            )
            manager = ToolProviderManager(runtime.tools, providers)
            report = manager.refresh()
            self.assertIn("domain_lookup", runtime.tools.names())
            self.assertIn("domain_lookup", report["providers"]["local"]["loaded"])

            conflicting = providers_from_config(
                {
                    "providers": [
                        {
                            "id": "one",
                            "tools": [{"name": "collision", "endpoint": "https://one.example.org/tool"}],
                        },
                        {
                            "id": "two",
                            "tools": [{"name": "collision", "endpoint": "https://two.example.org/tool"}],
                        },
                    ]
                }
            )
            protected_snapshot = set(runtime.tools.names())
            with self.assertRaises(ValueError):
                ToolProviderManager(runtime.tools, conflicting).refresh()
            self.assertEqual(set(runtime.tools.names()), protected_snapshot)


if __name__ == "__main__":
    unittest.main()
