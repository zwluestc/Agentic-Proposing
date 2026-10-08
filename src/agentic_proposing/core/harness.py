from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import json

from .context import ContextManager
from .hooks import HookBus, HookEvent
from .models import Event, RunState, Stage, ToolResult, jsonable
from .store import RunStore, WorkspaceLayout
from agentic_proposing.providers.base import TeacherProvider
from agentic_proposing.tools.base import ToolContext, ToolDispatcher


@dataclass(slots=True)
class HarnessResult:
    run_id: str
    status: str
    turns: int
    terminal_result: ToolResult | None
    state: RunState


class AgentHarness:
    """Claude-Code-style tool loop specialized for Agentic Proposing."""

    def __init__(
        self,
        provider: TeacherProvider,
        dispatcher: ToolDispatcher,
        context_manager: ContextManager,
        layout: WorkspaceLayout,
        hooks: HookBus | None = None,
        *,
        max_turns: int = 40,
        temperature: float = 0.2,
        max_tokens_per_turn: int = 8192,
        services: dict[str, Any] | None = None,
        action_observer: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.provider = provider
        self.dispatcher = dispatcher
        self.context_manager = context_manager
        self.layout = layout
        self.hooks = hooks or dispatcher.hooks
        self.max_turns = max_turns
        self.temperature = temperature
        self.max_tokens_per_turn = max_tokens_per_turn
        self.services = dict(services or {})
        self.action_observer = action_observer

    def run(self, objective: str, state: RunState | None = None) -> HarnessResult:
        self.layout.initialize()
        state = state or RunState.create(objective, self.layout.root)
        run_store = RunStore(self.layout, state.run_id)
        context = ToolContext(self.layout, run_store, state, self.services)
        state.budget.setdefault("max_turns", self.max_turns)
        run_store.save_state(state)
        self.hooks.emit(HookEvent.SESSION_START, {"state": state.to_dict()})
        run_store.events.append(Event("session_started", state.run_id, {"objective": objective}))
        terminal: ToolResult | None = None
        try:
            while state.turn < self.max_turns:
                state.turn += 1
                messages, tools = self.context_manager.materialize(state, run_store)
                self.hooks.emit(HookEvent.PRE_MODEL, {"turn": state.turn, "state": state.to_dict()})
                turn = self.provider.complete(
                    messages,
                    tools,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens_per_turn,
                )
                observation_stage = state.stage.value
                run_store.events.append(
                    Event(
                        "model_turn",
                        state.run_id,
                        {
                            "turn": state.turn,
                            "content": turn.content,
                            "tool_calls": jsonable(turn.tool_calls),
                            "usage": turn.usage,
                        },
                    )
                )
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": turn.content,
                }
                if turn.tool_calls:
                    assistant_message["tool_calls"] = [
                        {
                            "id": call.call_id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for call in turn.tool_calls
                    ]
                state.transcript.append(assistant_message)
                self.hooks.emit(HookEvent.POST_MODEL, {"turn": state.turn, "model_turn": jsonable(turn)})
                results: list[ToolResult] = []
                if not turn.tool_calls:
                    state.transcript.append(
                        {
                            "role": "user",
                            "content": (
                                "This harness does not accept a plain-text finish. Continue the loop, run "
                                "self_check_artifact on the exact TaskBundle when required, refine any failures, "
                                "then call submit_artifact."
                            ),
                        }
                    )
                    run_store.events.append(Event("non_terminal_text", state.run_id, {"turn": state.turn}))
                else:
                    results = self.dispatcher.execute(turn.tool_calls, context)
                    for result in results:
                        state.transcript.append(
                            {
                                "role": "tool",
                                "tool_call_id": result.call_id,
                                "name": result.tool_name,
                                "content": json.dumps(
                                    {
                                        "content": result.content,
                                        "data": result.data,
                                        "is_error": result.is_error,
                                        "artifact_refs": result.artifact_refs,
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )
                        if result.terminal and not result.is_error:
                            terminal = result
                            state.status = "completed"
                            state.stage = Stage.COMPLETE
                            break
                if self.action_observer is not None:
                    self.action_observer({
                        "run_id": state.run_id,
                        "turn": state.turn,
                        "stage": observation_stage,
                        "observation": messages,
                        "tools": tools,
                        "action": {
                            "content": turn.content,
                            "tool_calls": [
                                {"name": call.name, "arguments": call.arguments}
                                for call in turn.tool_calls
                            ],
                        },
                        "policy_raw": turn.raw if "completion_token_ids" in turn.raw else {},
                        "tool_results": [jsonable(result) for result in results],
                    })
                run_store.checkpoint(state)
                run_store.save_state(state)
                if terminal:
                    break
            if not terminal and state.status == "running":
                state.status = "max_turns"
                state.stage = Stage.FAILED
                run_store.events.append(Event("max_turns_reached", state.run_id, {"turn": state.turn}))
        except Exception as exc:
            state.status = "error"
            state.stage = Stage.FAILED
            run_store.events.append(
                Event("session_error", state.run_id, {"type": type(exc).__name__, "message": str(exc)})
            )
            run_store.save_state(state)
            self.hooks.emit(HookEvent.SESSION_END, {"state": state.to_dict(), "error": str(exc)})
            raise
        run_store.save_state(state)
        self.hooks.emit(HookEvent.SESSION_END, {"state": state.to_dict()})
        run_store.events.append(Event("session_ended", state.run_id, {"status": state.status}))
        return HarnessResult(state.run_id, state.status, state.turn, terminal, state)
