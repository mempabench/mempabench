"""The `mempabench` command.

`data pull` downloads the persona data from Hugging Face at the pinned
`DATASET_REVISION` and copies each file listed in `PERSONA_DATA_FILES` to the
same relative path under the checkout root, which is where the harness reads
it. Every other command hands its arguments unchanged to the
`main` of an existing entry point; `run` also exits with
`RESUMABLE_FAILURE_EXIT_CODE` when that `main` stops on a failure a rerun
resumes from.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from harness.failure_kind import FAILED_STEP_ID_ATTR, RESUMABLE_FAILURE_KINDS, classify_failure
from harness.run_layout import PROJECT_ROOT

DATASET_REPO = "mempabench/mempabench-personas"
#: The commit of `DATASET_REPO` this code reads.
DATASET_REVISION = "6d5fb9b2785bc9fb00c3e2c09920742043cd5a4f"

PERSONAS = ("user_a", "user_b", "user_c", "user_d")

#: Every persona data file (`<persona>/...`) of `DATASET_REPO` at the revision
#: on its first line; read by `load_persona_data_files`.
PERSONA_DATA_FILES = Path(__file__).with_name("persona_data_files.tsv")

#: Command -> the module whose `main()` runs with the remaining arguments.
ENTRY_POINTS = {
    "plan": "scripts.diagnostics.mint_run_plan_only",
    "run": "scripts.run_locked",
    "score": "scripts.score_completed_run",
    "self-report": "scripts.self_report_completed_run",
    "repeat-prepare": "scripts.prepare_probe_variance",
    "repeat-run": "scripts.run_probe_variance",
    "repeat-report": "scripts.report_probe_variance",
}
#: Commands whose own parser would otherwise use a default PA model.
PA_MODEL_REQUIRED = frozenset({"plan", "run"})
#: The exit code (EX_TEMPFAIL) of a `run` that stopped on a failure of one of
#: `RESUMABLE_FAILURE_KINDS`; rerunning the same command resumes it, and
#: `scripts/run_with_resume.sh` does that on this code only.
RESUMABLE_FAILURE_EXIT_CODE = 75
#: Starts the last stderr line of such a run:
#: `mempabench-run-stopped kind=<failure kind> step=<step id, or - outside a step>`.
RUN_STOPPED_LINE_PREFIX = "mempabench-run-stopped"

USAGE = """\
usage: mempabench <command> [arguments]

commands:
  data pull [--persona P]  download the persona data at the pinned revision
                           into this checkout
  plan            freeze a RunPlan and stop (--pa-model required)
  run             launch or resume one run (--pa-model required)
  score           score a finished run
  self-report     rate a scored run's final probes as its simulated user (PRS)
  repeat-prepare  prepare probe repeat rounds
  repeat-run      run probe repeat rounds
  repeat-report   report on probe repeat rounds

Arguments after the command go unchanged to its entry point;
`mempabench <command> --help` lists them."""


class PersonaDataPullError(RuntimeError):
    """The download lacks a file listed in `PERSONA_DATA_FILES`."""


def load_persona_data_files(path: Path) -> tuple[str, dict[str, list[str]]]:
    """Read a persona data file list: (dataset revision, persona -> paths).

    Line 1 is `revision<TAB><commit>`. Every other line is
    `<persona><TAB><path>`: the dataset holds the file at `<persona>/<path>`,
    and `data pull` puts it at `<checkout root>/<path>`.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    files: dict[str, list[str]] = {persona: [] for persona in PERSONAS}
    for line in lines[1:]:
        persona, relative = line.split("\t")
        files[persona].append(relative)
    return lines[0].split("\t")[1], files


def copy_persona_data(
    snapshot: Path,
    listed: Mapping[str, Sequence[str]],
    personas: Sequence[str],
    root: Path,
) -> int:
    """Copy each listed file `<snapshot>/<persona>/<path>` to `<root>/<path>`,
    overwriting what is there; return the number of files copied.

    Nothing is written when the snapshot lacks a listed file: the
    PersonaDataPullError names every missing one.
    """
    missing = [
        f"{persona}/{relative}"
        for persona in personas
        for relative in listed[persona]
        if not (snapshot / persona / relative).is_file()
    ]
    if missing:
        raise PersonaDataPullError(
            f"nothing was written under {root}; the snapshot {snapshot} lacks "
            f"{len(missing)} listed file(s):\n" + "\n".join(missing)
        )
    copied = 0
    for persona in personas:
        for relative in listed[persona]:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / persona / relative, target)
            copied += 1
    return copied


def pull(personas: Sequence[str]) -> int:
    from huggingface_hub import snapshot_download

    _, listed = load_persona_data_files(PERSONA_DATA_FILES)
    snapshot = Path(
        snapshot_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            revision=DATASET_REVISION,
            allow_patterns=[f"{persona}/**" for persona in personas],
        )
    )
    copied = copy_persona_data(snapshot, listed, personas, PROJECT_ROOT)
    print(f"{DATASET_REPO}@{DATASET_REVISION}: {copied} files copied under {PROJECT_ROOT}")
    return 0


def _data(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="mempabench data")
    actions = parser.add_subparsers(dest="action", required=True)
    pull_parser = actions.add_parser(
        "pull", help=f"download {DATASET_REPO} at the pinned revision into this checkout"
    )
    pull_parser.add_argument(
        "--persona",
        action="append",
        choices=PERSONAS,
        help="persona to pull; repeatable; default: all four",
    )
    args = parser.parse_args(argv)
    return pull(list(dict.fromkeys(args.persona or PERSONAS)))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if not args:
        print(USAGE, file=sys.stderr)
        return 2
    command, rest = args[0], args[1:]
    if command in ("-h", "--help"):
        print(USAGE)
        return 0
    if command == "data":
        return _data(rest)
    if command not in ENTRY_POINTS:
        print(f"mempabench: unknown command {command!r}\n\n{USAGE}", file=sys.stderr)
        return 2
    asks_help = "-h" in rest or "--help" in rest
    names_pa_model = any(arg == "--pa-model" or arg.startswith("--pa-model=") for arg in rest)
    if command in PA_MODEL_REQUIRED and not asks_help and not names_pa_model:
        print(f"mempabench {command}: --pa-model is required", file=sys.stderr)
        return 2
    sys.argv = [f"mempabench {command}", *rest]
    entry_point = importlib.import_module(ENTRY_POINTS[command])
    if command != "run":
        return entry_point.main()
    try:
        return entry_point.main()
    except Exception as error:
        kind = classify_failure(error)
        if kind not in RESUMABLE_FAILURE_KINDS:
            raise
        # Printed by the hook the interpreter uses for an uncaught exception;
        # only the exit code and the stop line after it differ.
        sys.excepthook(type(error), error, error.__traceback__)
        step = getattr(error, FAILED_STEP_ID_ATTR, None) or "-"
        print(f"{RUN_STOPPED_LINE_PREFIX} kind={kind} step={step}", file=sys.stderr, flush=True)
        return RESUMABLE_FAILURE_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
