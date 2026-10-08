"""Resumable, quota-balanced API collection of real harness trajectories."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any
import hashlib
import json
import os
import time

from agentic_proposing.config import create_provider, resolve_provider_config
from agentic_proposing.core.store import atomic_write_text
from agentic_proposing.providers.base import TeacherProvider
from agentic_proposing.runtime import build_runtime
from agentic_proposing.skills.dedup import fingerprint
from .data import read_jsonl, seed_objective, seed_run_state, write_jsonl
from .math_tasks import solve
from .prober import answers_match
from .math_parameters import suggest


class MeteredProvider(TeacherProvider):
    def __init__(self, provider, usage):
        self.provider, self.usage = provider, usage

    def complete(self, messages, tools, **kwargs):
        turn = self.provider.complete(messages, tools, **kwargs)
        self.usage["calls"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            self.usage[key] += int(turn.usage.get(key, 0))
        return turn


def witness_key(bundle):
    return hashlib.sha256(json.dumps(bundle["metadata"]["math_spec"],sort_keys=True,separators=(",",":")).encode()).hexdigest()


def collection_signature(raw,seeds):
    return hashlib.sha256(json.dumps({
        "teacher_model":raw["teacher"]["model"],"verifier_model":raw["verifier"]["model"],
        "seeds":seeds,"runtime":raw["runtime"],"target":int(raw["sft"]["target_accepted"]),
        "math_parameter_suggestions":bool(raw.get("math_parameter_suggestions",False)),
    },sort_keys=True).encode()).hexdigest()


def trajectory_defect(record):
    """Require real Skill use and an inspected, final-witness numerical execution."""
    transitions, bundle = record["transitions"], record["task_bundle"]
    if len(transitions) < 4:
        return "too_few_observed_actions"
    if not any(r["tool_name"] == "skill_load" and not r["is_error"] for t in transitions for r in t["tool_results"]):
        return "no_successful_skill_load"
    spec = bundle.get("metadata",{}).get("math_spec")
    if not isinstance(spec,dict) or not answers_match(bundle.get("answer"),solve(spec)["answer"]):
        return "invalid_final_math_witness"
    found = False
    for index,t in enumerate(transitions):
        calls = t["action"].get("tool_calls",[])
        for call in calls:
            if call["name"] != "skill_python" or call["arguments"].get("input") != spec:
                continue
            if call["arguments"].get("skill_id") not in bundle.get("skill_lineage",[]):
                continue
            if any(r["tool_name"] == "skill_python" and not r["is_error"]
                   and answers_match(r.get("data",{}).get("result",{}).get("answer"),bundle["answer"])
                   for r in t["tool_results"]):
                # The proposer must have received the execution result before final submission.
                found = index < len(transitions)-1
    return None if found else "no_inspected_final_witness_execution"


def collect_to_target(raw: dict[str,Any]) -> Path:
    from .runner import _prepare_workspace, _runtime_config
    for role in ("teacher","verifier"):
        provider = resolve_provider_config(raw[role])
        if not provider.get("api_key") and not os.environ.get(provider.get("api_key_env","OPENAI_API_KEY")):
            raise RuntimeError(f"Set {provider.get('api_key_env','OPENAI_API_KEY')} for {role}")
    workspace = _prepare_workspace(raw)
    settings = raw["sft"]
    output = Path(settings["data_file"]).resolve()
    rejection_path = output.with_suffix(".rejected.jsonl")
    failed_path = output.with_suffix(".failed.jsonl")
    overflow_path = output.with_suffix(".overflow.jsonl")
    progress_path = output.with_suffix(".progress.json")
    target = int(settings["target_accepted"])
    workers = int(settings.get("collection_workers",4))
    limit = int(settings.get("max_collection_attempts",target*3))
    if target < 1 or workers < 1 or limit < target:
        raise ValueError("Invalid collection target, workers or attempt budget")
    seeds = read_jsonl(raw["seeds"])
    grouped = {}
    for seed in seeds:
        grouped.setdefault(seed["category"],[]).append(seed)
    categories = sorted(grouped)
    quotas = {c:target//len(categories)+int(i<target%len(categories)) for i,c in enumerate(categories)}
    signature = collection_signature(raw,seeds)
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if previous and previous.get("configuration_fingerprint") != signature:
        raise ValueError("Collection configuration changed; use a new data_file to avoid mixing datasets")
    records = read_jsonl(output) if output.exists() and output.stat().st_size else []
    counts = Counter(r["category"] for r in records)
    questions = {fingerprint(r["task_bundle"]["question"]) for r in records}
    witnesses = {witness_key(r["task_bundle"]) for r in records}
    if len(questions) != len(records) or len(witnesses) != len(records):
        raise ValueError("Existing accepted data contains duplicates")
    attempts = int(previous.get("attempts",0))
    rejections = Counter(previous.get("rejections",{}))
    usage = Counter(previous.get("api_usage",{}))
    next_by_category = Counter(previous.get("attempts_by_category",{}))
    in_flight = Counter()
    started = time.monotonic()

    def progress(status):
        report = {"status":status,"configuration_fingerprint":signature,"target_accepted":target,
                  "accepted":sum(counts.values()),"attempts":attempts,"accepted_by_category":dict(counts),
                  "quota_by_category":quotas,"attempts_by_category":dict(next_by_category),
                  "rejections":dict(rejections),"api_usage":dict(usage),"workers":workers,
                  "elapsed_this_process_seconds":round(time.monotonic()-started,1),
                  "teacher_model":raw["teacher"]["model"],"verifier_model":raw["verifier"]["model"],
                  "data_file":str(output),"official_paper_data":False}
        atomic_write_text(progress_path,json.dumps(report,ensure_ascii=False,indent=2))
        return report

    def episode(seed,number):
        transitions, episode_usage = [], Counter()
        teacher = MeteredProvider(create_provider(raw["teacher"]),episode_usage)
        verifier = MeteredProvider(create_provider(raw["verifier"]),episode_usage)
        runtime = build_runtime(workspace,teacher,_runtime_config(raw),verifier_provider=verifier,
                                action_observer=transitions.append)
        objective = seed_objective(seed)+f" Instance index: {number}. Choose fresh parameters and presentation."
        if raw.get("math_parameter_suggestions", False):
            objective += (
                " Suggested fresh construction input (contains no supplied answer): "
                + json.dumps(suggest(seed["category"],number),sort_keys=True)
                + ". Use this input unless a real feasibility check requires revision. "
                  "The question must request ONLY the scalar result; put the short solution in rationale. "
                  "Do not add 'explain/prove' to the question. Omit rubric or set rubric=[]. "
                  "Include the loaded construction Skill ID in TaskBundle.skill_lineage. "
                  "Each content field should be a brief decision or checked conclusion, not a planning monologue. "
                  "Prefer 5-8 purposeful tool turns."
            )
        state = seed_run_state(seed,objective,workspace,runtime.registry)
        def failed_record(reason, *, status=None, error=None):
            secrets = [resolve_provider_config(raw[role]).get("api_key") for role in ("teacher", "verifier")]
            def redact(value):
                if isinstance(value, str):
                    for secret in secrets:
                        if secret:
                            value = value.replace(secret, "[REDACTED]")
                    return value
                if isinstance(value, list):
                    return [redact(item) for item in value]
                if isinstance(value, dict):
                    return {key: redact(item) for key, item in value.items()}
                return value
            row = {"trajectory_id":state.run_id,"seed_id":seed["id"],"category":seed["category"],
                   "objective":objective,"transitions":transitions,
                   "task_bundle":dict(state.working_artifact) if isinstance(state.working_artifact,dict) else {},
                   "valid":False,"failure_reason":reason,"teacher_model":raw["teacher"]["model"],
                   "verifier_model":raw["verifier"]["model"],"collection_attempt":number}
            if status is not None:
                row["run_status"] = status
            if error is not None:
                row["error_type"],row["error"] = error
            return redact(row)
        try:
            result = runtime.harness.run(objective,state)
            if result.status != "completed":
                reason = str(result.status)
                return None,{"run_id":result.run_id,"category":seed["category"],"reason":reason,
                             "failed_trajectory":failed_record(reason,status=reason)},episode_usage
            record = {"trajectory_id":result.run_id,"seed_id":seed["id"],"category":seed["category"],
                      "objective":objective,"transitions":transitions,"task_bundle":dict(result.state.working_artifact),
                      "valid":True,"teacher_model":raw["teacher"]["model"],"verifier_model":raw["verifier"]["model"],
                      "collection_attempt":number}
            defect = trajectory_defect(record)
            if defect:
                return None,{"run_id":result.run_id,"category":seed["category"],"reason":defect,
                             "failed_trajectory":dict(record,valid=False,failure_reason=defect)},episode_usage
            return record,None,episode_usage
        except Exception as exc:
            message = str(exc)
            for role in ("teacher","verifier"):
                secret = resolve_provider_config(raw[role]).get("api_key")
                if secret:
                    message = message.replace(secret,"[REDACTED]")
            error = (type(exc).__name__,message[:500])
            return None,{"run_id":state.run_id,"category":seed["category"],"reason":"episode_error",
                         "error_type":error[0],"error":error[1],
                         "failed_trajectory":failed_record("episode_error",error=error)},episode_usage
        finally:
            runtime.python_session_manager.close_run(state.run_id)

    progress("running")
    fatal_errors = 0
    api_blocked = False
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        while sum(counts.values()) < target:
            while not api_blocked and len(pending) < workers and attempts < limit:
                available = [c for c in categories if counts[c]+in_flight[c] < quotas[c]]
                if not available:
                    break
                category = min(available,key=lambda c:(counts[c]+in_flight[c],next_by_category[c],c))
                seed = grouped[category][next_by_category[category] % len(grouped[category])]
                next_by_category[category] += 1
                attempts += 1
                in_flight[category] += 1
                pending[pool.submit(episode,seed,attempts)] = category
            if not pending:
                break
            done,_ = wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            if not done:
                progress("blocked_api_errors" if api_blocked else "running")
                print(json.dumps({"accepted":sum(counts.values()),"target":target,"attempts_started":attempts,"in_flight":len(pending)}),flush=True)
                continue
            for future in done:
                category = pending.pop(future)
                in_flight[category] -= 1
                record,rejection,meter = future.result()
                usage.update(meter)
                if record:
                    qkey,wkey = fingerprint(record["task_bundle"]["question"]),witness_key(record["task_bundle"])
                    if qkey in questions or wkey in witnesses:
                        rejection = {"run_id":record["trajectory_id"],"category":category,"reason":"duplicate_question_or_construction"}
                        rejection["failed_trajectory"] = dict(record,valid=False,failure_reason=rejection["reason"])
                    elif counts[category] >= quotas[category]:
                        write_jsonl(overflow_path,[record],append=True)
                    else:
                        write_jsonl(output,[record],append=True)
                        counts[category] += 1
                        questions.add(qkey)
                        witnesses.add(wkey)
                        fatal_errors = 0
                if rejection:
                    rejections[rejection["reason"]] += 1
                    failed = rejection.pop("failed_trajectory",None)
                    if failed is not None:
                        write_jsonl(failed_path,[failed],append=True)
                    write_jsonl(rejection_path,[rejection],append=True)
                    fatal_errors = fatal_errors+1 if rejection["reason"] == "episode_error" else 0
                if fatal_errors >= max(workers,3):
                    api_blocked = True
                progress("blocked_api_errors" if api_blocked else "running")
                print(json.dumps({"accepted":sum(counts.values()),"target":target,"attempts_started":attempts,
                                  "category":category,"category_accepted":counts[category],
                                  "last_rejection":rejection["reason"] if rejection else None}),flush=True)
    complete = sum(counts.values()) == target
    report = progress("completed" if complete else "blocked_api_errors" if api_blocked else "attempt_budget_exhausted")
    if api_blocked and not complete:
        raise RuntimeError("Repeated API/runtime failures; progress preserved. Inspect rejection log before resuming.")
    if not complete:
        raise RuntimeError(f"Collected {report['accepted']}/{target}; attempt budget exhausted. Data is resumable.")
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return output
