from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import argparse
import json
import sys

from .bootstrap import bootstrap_demo_workspace
from .config import RuntimeConfig, create_provider, load_mapping
from .core.models import ModelTurn, ToolCall, jsonable
from .core.store import WorkspaceLayout, atomic_write_text
from .corpus.chunking import SemanticChunker
from .corpus.parsers import BenchmarkParser, DocumentParser
from .evolution import RecursiveEvolutionEngine
from .providers.scripted import ScriptedTeacher
from .runtime import build_runtime
from .skills.extraction import SkillExtractor
from .skills.management import deprecate_skill, fork_skill_version
from .skills.models import CompositeSpec
from .skills.registry import SkillRegistry


DEFAULT_CONFIG = {
    "provider": {
        "type": "openai_compatible",
        "base_url": "https://api.openai.com/v1",
        "model": "SET_A_SOTA_TEACHER_MODEL",
        "api_key_env": "OPENAI_API_KEY",
    },
    "harness": {"max_turns": 40, "temperature": 0.2, "max_tokens_per_turn": 8192},
    "permissions": {
        "allow_read_only": True,
        "allow_network": False,
        "allow_process": False,
        "allow_registry_writes": True,
        "allowed": [],
        "denied": [],
        "ask": [],
    },
    "context": {"max_context_chars": 100000, "max_transcript_chars": 45000},
    "sources": {"roots": []},
    "tools": {"max_parallel": 8, "max_inline_chars": 24000},
    "python": {"allowed_imports": [], "default_timeout_seconds": 20},
    "web": {
        "backend": "",
        "allowed_domains": [],
        "timeout_seconds": 30,
    },
    "external_tools": {"providers": [], "refresh_on_start": False},
    "skill_mining": {"enabled": True, "max_events": 300},
    "verification": {
        "contamination_questions": [],
        "require_self_check": True,
        "model_committee_enabled": False,
        "model_voters": 3,
        "second_audit": True,
    },
}


def _json_print(value: Any) -> None:
    print(json.dumps(jsonable(value), ensure_ascii=False, indent=2, sort_keys=True))


def _load_runtime(args: argparse.Namespace):
    config_path = Path(args.config or Path(args.workspace) / "config.json")
    config = RuntimeConfig.from_file(config_path)
    provider = create_provider(config.provider)
    return build_runtime(args.workspace, provider, config), config


def command_init(args: argparse.Namespace) -> int:
    layout = WorkspaceLayout(Path(args.workspace).resolve())
    layout.initialize()
    config_path = layout.root / "config.json"
    if not config_path.exists():
        atomic_write_text(config_path, json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2, sort_keys=True))
    if args.with_examples:
        bootstrap_demo_workspace(layout.root)
    _json_print({"workspace": str(layout.root), "config": str(config_path), "examples": args.with_examples})
    return 0


def command_demo(args: argparse.Namespace) -> int:
    bootstrap_demo_workspace(args.workspace)
    bundle = {
        "task_id": "demo-clearance-001",
        "task_type": "single_choice",
        "question": (
            "A synthetic case defines clearance as 100 divided by a measured value. "
            "The value changes from 4 at baseline to 5 at follow-up. By how many units did clearance decrease?"
        ),
        "answer": "5",
        "rationale": "Baseline clearance is 25 and follow-up clearance is 20, so the decrease is 5.",
        "rubric": [],
        "evidence": [],
        "skill_lineage": ["med.calc.two-stage-clearance", "task.choice.numeric-distractors"],
        "difficulty": {"estimated": 2},
        "metadata": {
            "knowledge_dependency": False,
            "options": ["3", "4", "5", "6"],
            "output_constraints": {},
        },
    }
    teacher = ScriptedTeacher(
        [
            ModelTurn(tool_calls=[ToolCall("skill_search", {"query": "temporal calculation numeric choice"})]),
            ModelTurn(tool_calls=[ToolCall("skill_load", {"skill_ids": bundle["skill_lineage"]})]),
            ModelTurn(
                tool_calls=[
                    ToolCall("python_eval", {"expression": "100/4"}),
                    ToolCall("python_eval", {"expression": "100/5"}),
                    ToolCall(
                        "working_state_edit",
                        {
                            "stage": "check",
                            "working_artifact": bundle,
                            "decision": "Verify both time-point calculations before submission.",
                        },
                    ),
                ]
            ),
            ModelTurn(tool_calls=[ToolCall("self_check_artifact", {"task_bundle": bundle})]),
            ModelTurn(tool_calls=[ToolCall("submit_artifact", {"task_bundle": bundle})]),
        ]
    )
    config = RuntimeConfig(
        harness={"max_turns": 8},
        permissions={"allow_read_only": True, "allow_registry_writes": True},
        skill_mining={"enabled": False},
        verification={"require_self_check": True},
    )
    runtime = build_runtime(args.workspace, teacher, config)
    result = runtime.harness.run("Run the offline Agentic Proposing demonstration")
    _json_print(
        {
            "run_id": result.run_id,
            "status": result.status,
            "turns": result.turns,
            "terminal": result.terminal_result.data if result.terminal_result else None,
            "events": str(runtime.layout.run_dir(result.run_id) / "events.jsonl"),
        }
    )
    return 0 if result.status == "completed" else 1


def command_ingest(args: argparse.Namespace) -> int:
    runtime, _ = _load_runtime(args)
    document = DocumentParser().parse(args.path, {"title": args.title or Path(args.path).stem})
    chunks = SemanticChunker(args.target_chars, args.overlap_chars).chunk(document)
    result = runtime.evidence_store.ingest(document, chunks)
    _json_print(result)
    return 0


def command_acquire(args: argparse.Namespace) -> int:
    runtime, _ = _load_runtime(args)
    extractor = SkillExtractor(runtime.provider, runtime.registry)
    outcomes = []
    if args.kind == "benchmark":
        items = BenchmarkParser().parse(args.path)
        for item in items[: args.limit or None]:
            outcomes.append(extractor.extract_from_qa(jsonable(item)))
    else:
        document = DocumentParser().parse(args.path)
        chunks = SemanticChunker(args.target_chars, args.overlap_chars).chunk(document)
        for chunk in chunks[: args.limit or None]:
            outcomes.append(extractor.extract_from_corpus(jsonable(chunk)))
    _json_print([asdict(outcome) for outcome in outcomes])
    return 0


def command_compose(args: argparse.Namespace) -> int:
    layout = WorkspaceLayout(Path(args.workspace).resolve())
    registry = SkillRegistry(layout)
    spec = CompositeSpec.from_dict(load_mapping(args.spec))
    destination = registry.stage_composite(spec)
    _json_print({"skill_id": spec.skill_id, "staged_path": str(destination)})
    return 0


def command_run(args: argparse.Namespace) -> int:
    runtime, _ = _load_runtime(args)
    result = runtime.harness.run(args.objective)
    _json_print(
        {
            "run_id": result.run_id,
            "status": result.status,
            "turns": result.turns,
            "terminal": result.terminal_result.data if result.terminal_result else None,
        }
    )
    return 0 if result.status == "completed" else 1


def command_evolve(args: argparse.Namespace) -> int:
    runtime, _ = _load_runtime(args)
    engine = RecursiveEvolutionEngine(runtime.provider, runtime.registry, runtime.harness)
    reports = engine.evolve(
        args.goal,
        args.rounds,
        args.compositions_per_round,
        args.replays_per_composition,
        args.min_replay_success_rate,
        not args.no_auto_promote,
    )
    _json_print([asdict(report) for report in reports])
    return 0


def command_skills(args: argparse.Namespace) -> int:
    layout = WorkspaceLayout(Path(args.workspace).resolve())
    registry = SkillRegistry(layout)
    if args.skills_command == "list":
        _json_print(registry.discovery_summaries())
    elif args.skills_command == "show":
        package = registry.get_package(args.skill_id)
        _json_print({"manifest": package.manifest.to_dict(), "body": package.body})
    elif args.skills_command == "lineage":
        _json_print(registry.lineage(args.skill_id, args.max_depth))
    elif args.skills_command == "promote":
        validation = load_mapping(args.validation)
        _json_print({
            "path": str(registry.promote(
                args.staged_name,
                validation,
                replace_existing=bool(args.replace_existing),
            ))
        })
    elif args.skills_command == "deprecate":
        _json_print({"path": str(deprecate_skill(registry, args.skill_id, args.reason))})
    elif args.skills_command == "version":
        _json_print({
            "path": str(fork_skill_version(
                registry,
                args.skill_id,
                args.new_version,
                args.change_summary,
            ))
        })
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agentic-proposing", description="Agentic Proposing reproduction harness")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Initialize a research workspace")
    init.add_argument("workspace")
    init.add_argument("--with-examples", action="store_true")
    init.set_defaults(handler=command_init)
    demo = sub.add_parser("demo", help="Run deterministic end-to-end demo")
    demo.add_argument("--workspace", required=True)
    demo.set_defaults(handler=command_demo)
    ingest = sub.add_parser("ingest", help="Parse and index a textbook/corpus")
    ingest.add_argument("path")
    ingest.add_argument("--workspace", required=True)
    ingest.add_argument("--config")
    ingest.add_argument("--title")
    ingest.add_argument("--target-chars", type=int, default=2600)
    ingest.add_argument("--overlap-chars", type=int, default=300)
    ingest.set_defaults(handler=command_ingest)
    acquire = sub.add_parser("acquire", help="Extract staged skills from benchmark items or corpus chunks")
    acquire.add_argument("kind", choices=["benchmark", "corpus"])
    acquire.add_argument("path")
    acquire.add_argument("--workspace", required=True)
    acquire.add_argument("--config")
    acquire.add_argument("--limit", type=int)
    acquire.add_argument("--target-chars", type=int, default=2600)
    acquire.add_argument("--overlap-chars", type=int, default=300)
    acquire.set_defaults(handler=command_acquire)
    compose = sub.add_parser("compose", help="Compile a composite skill spec into staging")
    compose.add_argument("spec")
    compose.add_argument("--workspace", required=True)
    compose.set_defaults(handler=command_compose)
    run = sub.add_parser("run", help="Run one proposing harness session")
    run.add_argument("objective")
    run.add_argument("--workspace", required=True)
    run.add_argument("--config")
    run.set_defaults(handler=command_run)
    evolve = sub.add_parser("evolve", help="Run recursive compose -> task -> mine rounds")
    evolve.add_argument("goal")
    evolve.add_argument("--workspace", required=True)
    evolve.add_argument("--config")
    evolve.add_argument("--rounds", type=int, default=2)
    evolve.add_argument("--compositions-per-round", type=int, default=3)
    evolve.add_argument("--replays-per-composition", type=int, default=2)
    evolve.add_argument("--min-replay-success-rate", type=float, default=0.8)
    evolve.add_argument("--no-auto-promote", action="store_true")
    evolve.set_defaults(handler=command_evolve)
    skills = sub.add_parser("skills", help="Inspect and manage the Skill Registry")
    skill_sub = skills.add_subparsers(dest="skills_command", required=True)
    skill_list = skill_sub.add_parser("list")
    show = skill_sub.add_parser("show")
    show.add_argument("skill_id")
    lineage = skill_sub.add_parser("lineage")
    lineage.add_argument("skill_id")
    lineage.add_argument("--max-depth", type=int, default=8)
    promote = skill_sub.add_parser("promote")
    promote.add_argument("staged_name")
    promote.add_argument("--validation", required=True)
    promote.add_argument("--replace-existing", action="store_true")
    deprecate = skill_sub.add_parser("deprecate")
    deprecate.add_argument("skill_id")
    deprecate.add_argument("--reason", required=True)
    version = skill_sub.add_parser("version")
    version.add_argument("skill_id")
    version.add_argument("new_version")
    version.add_argument("--change-summary", required=True)
    for skill_command in (skill_list, show, lineage, promote, deprecate, version):
        skill_command.add_argument("--workspace", required=True)
    skills.set_defaults(handler=command_skills)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

