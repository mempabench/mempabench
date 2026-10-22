#!/usr/bin/env bash
# Run `uv run mempabench run "$@"` and, while it exits 75, wait and run the
# identical command again. `mempabench run` exits 75 only when it stopped on a
# failure that a rerun resumes from (a model, provider, simulator, judge or
# memory-service failure: RESUMABLE_FAILURE_KINDS in harness/failure_kind.py),
# and then ends its stderr with the line
#   mempabench-run-stopped kind=<failure kind> step=<step id, or - outside a step>
# Any other exit code ends this script with that code.
#
#   MEMPABENCH_MAX_RESUMES              resumes after the first try (default 10)
#   MEMPABENCH_RESUME_WAIT_SECONDS      wait before a resume (default 300); it doubles
#                                       while the same step keeps failing
#   MEMPABENCH_RESUME_WAIT_MAX_SECONDS  the longest wait (default 3600)
#   MEMPABENCH_SAME_STEP_LIMIT          stop once one step failed this many tries in
#                                       a row (default 3)
#
# The run's stdout and stderr reach the terminal unchanged; a copy of the
# current try's stderr is kept in ${TMPDIR:-/tmp} to read the stop line and is
# removed when the try ends. At the end a table of all tries and a final status
# line go to stderr.
#
# Run it from the checkout with the arguments you would give `mempabench run`:
#   scripts/run_with_resume.sh --persona-id user_a --run-id my_run_01 ...
set -euo pipefail

readonly RESUMABLE_FAILURE_EXIT=75
readonly STOP_LINE_PREFIX='mempabench-run-stopped '
readonly STOP_LINE_PATTERN='^mempabench-run-stopped kind=([^ ]+) step=([^ ]+)$'

refuse() {
  echo "run_with_resume: $*" >&2
  exit 2
}

require_integer() {  # <variable name> <value> <minimum: 0 or 1>
  if [[ ! "$2" =~ ^[0-9]+$ ]] || (( 10#$2 < $3 )); then
    if (( $3 == 0 )); then
      refuse "$1 must be a non-negative integer, got '$2'"
    fi
    refuse "$1 must be a positive integer, got '$2'"
  fi
}

max_resumes="${MEMPABENCH_MAX_RESUMES-10}"
base_wait="${MEMPABENCH_RESUME_WAIT_SECONDS-300}"
max_wait="${MEMPABENCH_RESUME_WAIT_MAX_SECONDS-3600}"
same_step_limit="${MEMPABENCH_SAME_STEP_LIMIT-3}"
require_integer MEMPABENCH_MAX_RESUMES "$max_resumes" 0
require_integer MEMPABENCH_RESUME_WAIT_SECONDS "$base_wait" 0
require_integer MEMPABENCH_RESUME_WAIT_MAX_SECONDS "$max_wait" 0
require_integer MEMPABENCH_SAME_STEP_LIMIT "$same_step_limit" 1
max_resumes=$((10#$max_resumes))
base_wait=$((10#$base_wait))
max_wait=$((10#$max_wait))
same_step_limit=$((10#$same_step_limit))

# argparse takes any unambiguous prefix of an option, so a prefix of at least
# four characters (before any "=") counts as the flag itself.
for argument in "$@"; do
  name="${argument%%=*}"
  for flag in --overwrite --regenerate-plan; do
    if (( ${#name} >= 4 )) && [[ "$flag" == "$name"* ]]; then
      refuse "refusing '$argument' (taken as $flag): every resume repeats the arguments," \
        "so a resume would restart the run; for a fresh run use a new run id" \
        "without --overwrite or --regenerate-plan"
    fi
  done
done

timestamp() { date '+%Y-%m-%dT%H:%M:%S%z'; }

rows=()          # one summary row per try
final_status=""  # the last line printed at exit
stderr_copy=""   # the current try's stderr copy, removed when the try ends

add_row() {  # try started ended exit kind step next_wait
  rows+=("$(printf '%5s  %-24s  %-24s  %4s  %-22s  %-10s  %s' "$@")")
}

on_exit() {
  local status=$?
  if [[ -n "$stderr_copy" ]]; then
    rm -f "$stderr_copy"
  fi
  if (( ${#rows[@]} > 0 )); then
    {
      echo "run_with_resume summary:"
      printf '%5s  %-24s  %-24s  %4s  %-22s  %-10s  %s\n' \
        try started ended exit kind step next_wait_s
      printf '%s\n' "${rows[@]}"
      echo "${final_status:-run_with_resume: stopped with exit $status}"
    } >&2
  fi
}
trap on_exit EXIT

exec 3>&1  # the run's stdout bypasses the stderr pipe below

try=0
previous_step=""
same_step_failures=0
while true; do
  try=$((try + 1))
  started="$(timestamp)"
  stderr_copy="$(mktemp "${TMPDIR:-/tmp}/run_with_resume.XXXXXX")"
  # stdout goes straight to fd 3; stderr goes through tee, which ignores
  # Ctrl-C so the run's graceful-stop output still reaches the terminal.
  set +e
  uv run mempabench run "$@" 2>&1 1>&3 3>&- \
    | (trap '' INT TERM; exec tee "$stderr_copy" >&2 3>&-)
  pipe_status=("${PIPESTATUS[@]}")
  set -e
  status=${pipe_status[0]}
  ended="$(timestamp)"
  stop_line=""
  if (( status == RESUMABLE_FAILURE_EXIT )); then
    # The last stop line, not just the last line: shutdown noise may follow it.
    stop_line="$(grep -a "^$STOP_LINE_PREFIX" "$stderr_copy" | tail -n 1 || true)"
  fi
  rm -f "$stderr_copy"
  stderr_copy=""
  try_line="run_with_resume: try $try ended at $ended with exit $status"

  if (( status != RESUMABLE_FAILURE_EXIT )); then
    add_row "$try" "$started" "$ended" "$status" - - -
    echo "$try_line" >&2
    final_status="run_with_resume: finished at try $try with exit $status"
    exit "$status"
  fi
  if [[ ! "$stop_line" =~ $STOP_LINE_PATTERN ]]; then
    add_row "$try" "$started" "$ended" "$status" '?' '?' -
    echo "$try_line" >&2
    final_status="run_with_resume: try $try exited $status without a '${STOP_LINE_PREFIX}kind=... step=...' line, so where it stopped is unknown; not resuming, stopping with exit $status"
    exit "$status"
  fi
  kind="${BASH_REMATCH[1]}"
  step="${BASH_REMATCH[2]}"
  try_line="$try_line (kind=$kind step=$step)"

  if [[ "$step" != "-" && "$step" == "$previous_step" ]]; then
    same_step_failures=$((same_step_failures + 1))
  else
    same_step_failures=1
  fi
  previous_step="$step"
  if [[ "$step" != "-" ]] && (( same_step_failures >= same_step_limit )); then
    add_row "$try" "$started" "$ended" "$status" "$kind" "$step" -
    echo "$try_line" >&2
    final_status="run_with_resume: step $step failed $same_step_failures tries in a row (kind=$kind); retrying is not helping, stopping with exit $status"
    exit "$status"
  fi
  if (( try > max_resumes )); then
    add_row "$try" "$started" "$ended" "$status" "$kind" "$step" -
    echo "$try_line" >&2
    final_status="run_with_resume: no resumes left (MEMPABENCH_MAX_RESUMES=$max_resumes), stopping with exit $status"
    exit "$status"
  fi

  # The base wait, doubled for each earlier failure in a row at this step.
  wait_seconds=$base_wait
  for (( doubling = 1; doubling < same_step_failures; doubling++ )); do
    if (( wait_seconds > max_wait / 2 )); then
      wait_seconds=$max_wait
      break
    fi
    wait_seconds=$((wait_seconds * 2))
  done
  if (( wait_seconds > max_wait )); then
    wait_seconds=$max_wait
  fi
  add_row "$try" "$started" "$ended" "$status" "$kind" "$step" "$wait_seconds"
  echo "$try_line; resume $try of $max_resumes in $wait_seconds s" >&2
  sleep "$wait_seconds"
done
