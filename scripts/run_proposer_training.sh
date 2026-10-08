#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
stage="${1:-sft}"
if [[ $# -gt 0 ]]; then shift; fi

export PYTHONPATH="$repo_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Help and input inspection require neither CUDA nor training dependencies.
for arg in "$@"; do
  if [[ "$arg" == "--dry-run" || "$arg" == "--help" || "$arg" == "-h" ]]; then
    exec "${PROPOSER_PYTHON:-python3}" -m agentic_proposing.training.runner "$stage" "$@"
  fi
done

if [[ -z "${PROPOSER_PYTHON:-}" ]]; then
  if [[ ! -x .venv-proposer/bin/python ]]; then
    python3 -m venv .venv-proposer
  fi
  PROPOSER_PYTHON="$repo_dir/.venv-proposer/bin/python"
fi

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  if [[ "$stage" == "collect-sft" ]]; then
    "$PROPOSER_PYTHON" -m pip install -e .
  else
    "$PROPOSER_PYTHON" -m pip install -e '.[training]'
  fi
fi
exec "$PROPOSER_PYTHON" -m agentic_proposing.training.runner "$stage" "$@"
