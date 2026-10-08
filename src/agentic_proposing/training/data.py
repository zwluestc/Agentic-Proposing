from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import json


ACTION_PROTOCOL = (
    "Return exactly one JSON object with keys content (a concise, observable decision) "
    "and tool_calls (an array of objects with name and arguments). "
    "Use an empty array when there is no tool call. Do not emit hidden reasoning. "
    "To finish, call submit_artifact through tool_calls."
)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"No records in {path}")
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]], *, append: bool = False) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a" if append else "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def action_text(action: dict[str, Any]) -> str:
    calls = action.get("tool_calls", [])
    return json.dumps(
        {
            "content": str(action.get("content", "")),
            "tool_calls": [
                {"name": str(call["name"]), "arguments": dict(call.get("arguments", {}))}
                for call in calls
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def policy_messages(observation: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Encode native tool messages into an ordinary chat template shared by SFT and RL."""
    messages: list[dict[str, str]] = []
    for message in observation:
        role = str(message.get("role", "user"))
        content = str(message.get("content") or "")
        if role == "assistant" and message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                function = call.get("function", {})
                arguments = function.get("arguments", "{}")
                try:
                    arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    arguments = {"unparsed": str(arguments)}
                calls.append({"name": function.get("name", ""), "arguments": arguments})
            content = action_text({"content": content, "tool_calls": calls})
        elif role == "tool":
            content = f"Tool result ({message.get('name', 'tool')}): {content}"
            role = "user"
        elif role not in {"system", "user", "assistant"}:
            role = "user"
        messages.append({"role": role, "content": content})
    tool_contract = json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
    protocol = f"\n\n# Policy action protocol\n{ACTION_PROTOCOL}\nAvailable tools: {tool_contract}"
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] += protocol
    else:
        messages.insert(0, {"role": "system", "content": protocol})
    return messages


def encode_prompt(tokenizer: Any, observation: list[dict[str, Any]], tools: list[dict[str, Any]],
                  max_prompt_tokens: int) -> list[int]:
    messages = policy_messages(observation, tools)
    while True:
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                  enable_thinking=False)
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)
        if len(token_ids) <= max_prompt_tokens or len(messages) <= 2:
            break
        del messages[2]
    if len(token_ids) > max_prompt_tokens:
        raise ValueError(
            "System prompt, Skill metadata, and current state exceed max_prompt_tokens; "
            "raise the context limit or reduce the active Skill/tool catalog"
        )
    return token_ids


def encode_sft_example(tokenizer: Any, transition: dict[str, Any], max_length: int) -> dict[str, list[int]]:
    completion = action_text(dict(transition["action"]))
    completion_ids = tokenizer.encode(completion, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        completion_ids.append(tokenizer.eos_token_id)
    if len(completion_ids) >= max_length:
        raise ValueError("A demonstration action exceeds the configured sequence length")
    prompt_ids = encode_prompt(
        tokenizer, transition["observation"], transition["tools"],
        max_length - len(completion_ids),
    )
    return {
        "input_ids": prompt_ids + completion_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(completion_ids)),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def seed_objective(seed: dict[str, Any]) -> str:
    objective = str(seed.get("objective", "")).strip()
    if not objective:
        raise ValueError("Each seed needs a nonempty objective")
    return objective + (
        " Select and use relevant Skills, verify intermediate calculations with tools, "
        "and submit one novel TaskBundle with a unique answer. "
        "Record only concise observable decisions."
    )


def seed_run_state(seed: dict[str, Any], objective: str, workspace: str | Path, registry: Any):
    from agentic_proposing.core.models import RunState

    selected = [str(item) for item in seed.get("skill_ids", [])]
    missing = [skill_id for skill_id in selected if not registry.has(skill_id)]
    if missing:
        raise ValueError(f"Seed references Skills absent from the active library: {missing}")
    state = RunState.create(objective, workspace)
    state.active_skill_ids = selected
    return state
