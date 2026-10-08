#!/usr/bin/env python3
"""Use local/env credentials or prompt; collect or resume math SFT data."""
from pathlib import Path
import argparse
import getpass
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from agentic_proposing.config import load_mapping, resolve_provider_config
from agentic_proposing.training.collection import collect_to_target

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/proposer_training.math.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    raw = load_mapping(args.config)
    if not all(resolve_provider_config(raw[role]).get("api_key") for role in ("teacher", "verifier")):
        os.environ["ENVMUX_API_KEY"] = getpass.getpass("EnvMux key (not saved): ")
    collect_to_target(raw)
