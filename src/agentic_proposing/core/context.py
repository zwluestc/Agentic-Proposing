from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
import json

from .hooks import HookBus, HookEvent
from .models import Event, RunState, Stage
from .store import RunStore
from agentic_proposing.tools.base import ToolRegistry


class SkillContextSource(Protocol):
    def discovery_summaries(self, skill_ids: list[str] | None = None) -> list[dict[str, Any]]: ...
    def load_for_context(self, skill_id: str, max_chars: int) -> str: ...


BASE_SYSTEM_PROMPT = """You are the Teacher/Proposer inside an Agentic Proposing research harness.

Your objective is to produce a logically sound, evidence-grounded, verifiable TaskBundle. Work through Discover, Select, Compose, Draft, Check, Refine, and Finalize. These are revisitable phases, not a rigid one-way state machine.

Operating rules:
1. Use tool_search to discover and progressively load deferred capabilities; do not assume every schema is already in context.
2. Discover and load relevant skills. Keep reusable construction skills separate from task content and source evidence.
3. Use workspace_glob/read/grep for local material, web_search for pointers, evidence_ingest_url or corpus_ingest for provenance-preserving acquisition, then evidence_search/fetch for exact support.
4. Use a persistent python_session when analysis needs multiple dependent cells; use python_eval only for small isolated calculations. Treat the Python worker as a restricted convenience environment, not a hostile-code security boundary.
5. Save material progress with working_state_edit or artifact_write so it survives compaction. Full oversized tool results are stored as artifacts; follow artifact references when needed.
6. Run self_check_artifact on the exact candidate before final submission. If any check fails, revise and run it again; a changed candidate invalidates the previous check when freshness enforcement is enabled.
7. Never claim completion in plain text. submit_artifact is the only successful terminal action. If it rejects, use the structured failures and continue refining.
8. Do not expose private chain-of-thought. Record concise decisions, assumptions, checks, and tool-grounded conclusions.
"""


@dataclass(slots=True)
class ContextConfig:
    max_context_chars: int = 100_000
    max_transcript_chars: int = 45_000
    max_skill_chars_each: int = 12_000
    max_skill_total_chars: int = 30_000
    keep_recent_messages: int = 14


class ContextManager:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        skill_source: SkillContextSource | None = None,
        hooks: HookBus | None = None,
        config: ContextConfig | None = None,
    ):
        self.tool_registry = tool_registry
        self.skill_source = skill_source
        self.hooks = hooks or HookBus()
        self.config = config or ContextConfig()

    def _skill_context(self, state: RunState) -> str:
        if not self.skill_source:
            return ""
        chunks: list[str] = []
        remaining = self.config.max_skill_total_chars
        for skill_id in state.loaded_skill_ids:
            if remaining <= 0:
                break
            text = self.skill_source.load_for_context(
                skill_id, min(self.config.max_skill_chars_each, remaining)
            )
            chunks.append(f"\n## Loaded Skill: {skill_id}\n{text}")
            remaining -= len(text)
        return "".join(chunks)

    def _state_context(self, state: RunState) -> str:
        payload = {
            "run_id": state.run_id,
            "objective": state.objective,
            "stage": state.stage.value,
            "turn": state.turn,
            "active_skill_ids": state.active_skill_ids,
            "loaded_skill_ids": state.loaded_skill_ids,
            "composition_graph": state.composition_graph,
            "working_artifact": state.working_artifact,
            "evidence_refs": state.evidence_refs[-20:],
            "failed_checks": state.failed_checks[-10:],
            "artifact_refs": state.artifact_refs[-20:],
            "budget": state.budget,
            "runtime_metadata": {
                key: value for key, value in state.metadata.items()
                if key in {"self_check_fingerprint", "self_check_report", "python_sessions"}
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)

    def compact(self, state: RunState, run_store: RunStore) -> None:
        reserved = 30_000 + self.config.max_skill_total_chars
        effective_transcript_limit = min(
            self.config.max_transcript_chars,
            max(4_000, self.config.max_context_chars - reserved),
        )
        if len(json.dumps(state.transcript, ensure_ascii=False)) <= effective_transcript_limit:
            return
        self.hooks.emit(HookEvent.PRE_COMPACT, {"state": state.to_dict()})
        transcript = list(state.transcript)
        cut = max(0, len(transcript) - self.config.keep_recent_messages)
        while cut < len(transcript) and transcript[cut].get("role") == "tool":
            cut += 1
        recent = transcript[cut:]
        while recent and len(json.dumps(recent, ensure_ascii=False)) > effective_transcript_limit:
            recent.pop(0)
            while recent and recent[0].get("role") == "tool":
                recent.pop(0)
        old = transcript[: len(transcript) - len(recent)]
        tool_facts: list[str] = []
        for message in old:
            if message.get("role") == "tool":
                content = str(message.get("content", ""))
                tool_facts.append(content[:500])
        state.compact_summary = (
            f"Earlier turns compacted. Stage={state.stage.value}; "
            f"active_skills={state.active_skill_ids}; failed_checks={state.failed_checks[-5:]}; "
            f"selected_tool_facts={tool_facts[-8:]}"
        )
        state.transcript = recent
        run_store.events.append(
            Event("context_compacted", state.run_id, {"removed_messages": len(old)})
        )

    def materialize(self, state: RunState, run_store: RunStore) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.compact(state, run_store)
        system = BASE_SYSTEM_PROMPT
        if self.skill_source:
            summaries = self.skill_source.discovery_summaries()
            summaries.sort(key=lambda item:(item["skill_id"] not in state.active_skill_ids,item["skill_id"]))
            encoded = json.dumps(summaries,ensure_ascii=False,sort_keys=True)
            if len(encoded) > 18_000:
                summaries = [{key:item[key] for key in ("skill_id","version","status")}
                             | {"description":item.get("description","")[:160]}
                             for item in summaries]
                # Keep valid JSON rather than cutting through a metadata entry.
                kept = []
                for item in summaries:
                    if len(json.dumps(kept+[item],ensure_ascii=False)) > 18_000:
                        break
                    kept.append(item)
                encoded = json.dumps(kept,ensure_ascii=False,sort_keys=True)
                system += "\nMetadata is abbreviated; use skill_search/skill_load for full contracts."
                if len(kept) < len(summaries):
                    system += f" {len(summaries)-len(kept)} additional Skills are discoverable through skill_search."
            system += "\n\n# Discoverable Skill Metadata\n" + encoded
        system += self._skill_context(state)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": "# Current Working State\n" + self._state_context(state),
            },
        ]
        if state.compact_summary:
            messages.append({"role": "user", "content": "# Compacted History\n" + state.compact_summary})
        messages.extend(state.transcript)
        tools = self.tool_registry.model_tools(state.enabled_tool_names)
        return messages, tools
