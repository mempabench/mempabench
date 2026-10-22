#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-.}"

# Defaults:
#   PERSONA_ID=user_a
#   ACC_RANGE=1-10
#   MEMORY_CONDITION=nanobot_file_memory
#   PA_MODEL=<configs/pa_models.yaml default>
#
# Examples:
#   ./Run_bash/run_real_pa_session.sh
#   MEMORY_CONDITION=no_memory ./Run_bash/run_real_pa_session.sh --overwrite
#   ACC_RANGE=1-3 ./Run_bash/run_real_pa_session.sh
#   PERSONA_ID=user_b ./Run_bash/run_real_pa_session.sh

PERSONA_ID="${PERSONA_ID:-user_a}"
ACC_RANGE="${ACC_RANGE:-1-10}"
MEMORY_CONDITION="${MEMORY_CONDITION:-nanobot_file_memory}"

if [[ ! "${ACC_RANGE}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
  echo "ACC_RANGE must use START-END with positive integers, for example 1-10" >&2
  exit 2
fi

ACC_START=$((10#${BASH_REMATCH[1]}))
ACC_END=$((10#${BASH_REMATCH[2]}))
if (( ACC_START < 1 || ACC_END < ACC_START )); then
  echo "ACC_RANGE must start at 1 or greater and END must be at least START" >&2
  exit 2
fi

DEFAULT_PA_MODEL="$(python3 - <<'PY'
import yaml
from pathlib import Path
cfg = yaml.safe_load(Path("configs/pa_models.yaml").read_text(encoding="utf-8"))
print(cfg["default"])
PY
)"
PA_MODEL="${PA_MODEL:-${DEFAULT_PA_MODEL}}"

exec uv run python scripts/run_orchestrator.py \
  --persona-id "${PERSONA_ID}" \
  --acc-range "${ACC_RANGE}" \
  --memory-condition "${MEMORY_CONDITION}" \
  --pa-model "${PA_MODEL}" \
  "$@"
