#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo_dir/scripts/run_proposer_training.sh" mgpo \
  --config "$repo_dir/configs/proposer_training.math.grpo100.json" "$@"
