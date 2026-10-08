#!/usr/bin/env python3
"""Explicitly migrate v1 math Skills and collection quotas to the 40-family library.

Old Skill packages and progress are archived. Accepted JSONL bytes are preserved;
each carried record is rechecked against exact grading and its original API report.
No model vote or training trajectory is synthesized by this migration.
"""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from agentic_proposing.config import load_mapping
from agentic_proposing.core.store import atomic_write_text
from agentic_proposing.training.collection import collection_signature,trajectory_defect
from agentic_proposing.training.data import read_jsonl
from agentic_proposing.training.runner import _prepare_workspace


def upgrade(raw):
    source = Path(raw["skill_library"])
    workspace = Path(raw["workspace"])
    data = Path(raw["sft"]["data_file"])
    progress_path = data.with_suffix(".progress.json")
    old_progress = load_mapping(progress_path) if progress_path.exists() else {}
    records = read_jsonl(data) if data.exists() and data.stat().st_size else []
    before = hashlib.sha256(data.read_bytes()).hexdigest() if data.exists() else None
    seeds = read_jsonl(raw["seeds"])
    categories = sorted({s["category"] for s in seeds})
    for record in records:
        if record["category"] not in categories or trajectory_defect(record):
            raise ValueError("Existing trajectory failed migration validation")
        committee = [c for c in record["task_bundle"]["verification"]["gates"]["checks"]
                     if c["name"] == "model_verifier_committee"]
        if len(committee) != 1 or not committee[0]["passed"] or not committee[0]["details"]["accepted"]:
            raise ValueError("Carried trajectory lacks an accepted original API verifier report")
    archived = []
    for manifest in source.glob("*/manifest.yaml"):
        package = manifest.parent
        destination = workspace / "skills/atomic" / package.name
        if not destination.exists():
            continue
        existing = load_mapping(destination / "manifest.yaml")
        incoming = load_mapping(manifest)
        if existing == incoming:
            continue
        if existing["skill_id"] != incoming["skill_id"] or existing["version"] != "1.0.0" or incoming["version"] != "2.0.0":
            raise ValueError(f"Unexpected Skill change: {destination}")
        archive = workspace / "skills/deprecated" / f"{package.name}@1.0.0"
        if archive.exists():
            raise FileExistsError(archive)
        archive.parent.mkdir(parents=True,exist_ok=True)
        replacement = destination.with_name(destination.name+".upgrade-tmp")
        shutil.copytree(package,replacement)
        os.replace(destination,archive)
        try:
            os.replace(replacement,destination)
        except Exception:
            os.replace(archive,destination)
            raise
        archived.append(str(archive))
    _prepare_workspace(raw)
    signature = collection_signature(raw,seeds)
    report_path = data.with_suffix(".library-v2-migration.json")
    if old_progress and old_progress.get("configuration_fingerprint") != signature:
        snapshot = data.with_suffix(".progress.v1.json")
        if snapshot.exists():
            raise FileExistsError(snapshot)
        shutil.copyfile(progress_path,snapshot)
        target = int(raw["sft"]["target_accepted"])
        counts = Counter(r["category"] for r in records)
        quotas = {c:target//len(categories)+int(i<target%len(categories)) for i,c in enumerate(categories)}
        if any(counts[c] > quotas[c] for c in counts):
            raise ValueError("Carried data exceeds new category quotas")
        progress = dict(old_progress,configuration_fingerprint=signature,accepted=len(records),
                        accepted_by_category=dict(counts),quota_by_category=quotas,
                        library_version="2.0.0",migration_report=str(report_path))
        atomic_write_text(progress_path,json.dumps(progress,indent=2)+"\n")
        report = {"library_version":"2.0.0","constructor_skills":40,"supporting_skills":10,
                  "seed_objectives":len(seeds),"old_configuration_fingerprint":old_progress.get("configuration_fingerprint"),
                  "new_configuration_fingerprint":signature,"carried_trajectories":[r["trajectory_id"] for r in records],
                  "data_sha256":before,"data_bytes_preserved":True,"archived_skills":archived,
                  "validation":"Exact construction, actual tool trace and original accepted API committee report; no new API votes."}
        atomic_write_text(report_path,json.dumps(report,indent=2)+"\n")
    assert before == (hashlib.sha256(data.read_bytes()).hexdigest() if data.exists() else None)
    return {"workspace":str(workspace),"constructor_skills":40,"supporting_skills":10,
            "seeds":len(seeds),"preserved_trajectories":len(records),"archived_skills":len(archived)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/proposer_training.math.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    print(json.dumps(upgrade(load_mapping(args.config))))
