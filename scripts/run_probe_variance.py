"""Run independent Probe repeats with one memory owner per setting.

The JSON manifest (``mempa_probe_variance/1``) contains absolute
``workspace_root`` and ``state_root`` paths and ``settings``. Each setting has
an ``id``, a ``source`` reference ``{run_dir}``, for a reference arm an
``arm`` ``{memory_condition, pa_model}``, and one repeat per round of
``harness.probe_repeat_source.repeat_rounds`` (``repeat.rounds`` in
configs/run_config.yaml, plus round 1 for an arm). Each repeat has ``repeat``,
``argv`` (the original run_orchestrator arguments, including --plan) and the
identical ``source`` reference. How many pre-event and final Probes a round
runs is ``probe_variance`` in configs/run_config.yaml. Full frozen plans are
supplied by preparation; this entrypoint never rewrites them. Validation is the
one place a repeat's plan and GT copy are compared with its source's.

Scoring is handed off through durable round_complete.json records. Only
``--execute --score`` launches the original scorer with its explicitly enabled
variance retry policy, at most ``--score-concurrency`` jobs. An interrupted job
may reenter once per dispatcher restart; the scorer's persistent request ledger
holds uncertain paid requests while allowing other unrequested questions onward.
A round whose exhausted Probes are N/A is scored with
``score_completed_run --not-applicable-exhausted``.

Without --execute this command only validates and prints a plan.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
from typing import Any
from types import SimpleNamespace

import fcntl
import yaml

from harness.probe_slots import active_batch_slots, batch_capacity, configure_batch_slots
from harness.run_config import load_run_config
from harness.run_layout import PRODUCER_LOCKS_SEGMENT
from scripts.run_locked import producer_lock, workspace_lock

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mempa_probe_variance/1"
PROBE_KINDS = {"pre_event_probe", "final_probe"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def absolute(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"Manifest paths must be absolute: {value}")
    return path.resolve()


def key(setting: dict) -> str:
    # Stable across separate manifests: two dispatchers cannot own one source
    # setting (one arm of it) simply by choosing different labels or state directories.
    identity = str(absolute(setting["source"]["run_dir"]))
    if setting.get("arm") is not None:
        identity = json.dumps([identity, setting["arm"]["memory_condition"], setting["arm"]["pa_model"]])
    return hashlib.sha256(identity.encode()).hexdigest()


def setting_dir(manifest: dict, setting: dict) -> Path:
    return Path(manifest["state_root"]) / "settings" / key(setting)


def guard_path(manifest: dict, setting: dict) -> Path:
    return Path(manifest["workspace_root"]) / ".probe-variance-locks" / (key(setting) + ".lock")


@contextmanager
def claim(path: Path):
    """A stable flock inode is the authority; PID files alone are not claims."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def occupied(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False


def parse_repeat(repeat: dict):
    from scripts.run_orchestrator import _build_arg_parser
    argv = repeat["argv"]
    if not isinstance(argv, list) or not all(isinstance(arg, str) for arg in argv):
        raise ValueError("repeat argv must be a list of exact runner arguments")
    args = _build_arg_parser().parse_args(argv)
    if (not args.plan or args.overwrite or args.regenerate_plan
            or args.only_steps or args.force_ix):
        raise ValueError("Repeats require a complete frozen --plan without reset, selection or forced IX")
    path = absolute(args.plan)
    rerun = [s.strip() for s in (args.rerun_probe_steps or "").split(",") if s.strip()]
    raised = {item.partition("=")[0].strip() for item in (args.pa_max_tokens_for_probe or [])}
    if not raised <= set(rerun):
        raise ValueError("A raised PA cap applies only to a Probe being rerun")
    return args, path, yaml.safe_load(path.read_text())


def require_retryable(run: Path, args) -> None:
    """Before executing: a rerun Probe must have failed with attempts left.

    A failed Probe that has not used its three attempts is
    retried normally; successful and exhausted Probes are never rerun.
    """
    ledger = read(run / "ledger.json", {}).get("steps", {})
    for step_id in [s.strip() for s in (args.rerun_probe_steps or "").split(",") if s.strip()]:
        from harness.probe_na import ATTEMPT_LIMIT, TOTAL_ATTEMPT_LIMIT, counted_failures
        record = read(run / "probe_attempts" / f"{step_id}.json", {})
        attempts = record.get("attempts", [])
        start = record.get("reopened", {}).get("after_attempt", 0)
        # "running" is a Probe left mid-flight by an interrupted worker; its
        # attempt record still decides whether a retry is allowed.
        if (ledger.get(step_id, {}).get("status") not in {"failed", "running"} or counted_failures(attempts[start:]) >= ATTEMPT_LIMIT
                or len(attempts) >= TOTAL_ATTEMPT_LIMIT or any(a.get("status") != "failed" for a in attempts)):
            raise ValueError(f"Only a failed Probe with attempts left may be rerun: {step_id}")


def validate_manifest(path: Path) -> dict:
    data = read(path)
    if data.get("schema") != SCHEMA:
        raise ValueError("Unknown Probe variance manifest schema")
    workspace = absolute(data["workspace_root"])
    absolute(data["state_root"])
    from harness.probe_repeat_source import load_repeat_source, repeat_base_plan, repeat_rounds
    from harness.run_plan import RunPlan
    scope = load_run_config().probe_variance
    identities, labels, run_ids, run_dirs = set(), set(), set(), set()
    if not data["settings"]:
        raise ValueError("Manifest has no settings")
    for setting in data["settings"]:
        if not isinstance(setting["id"], str) or not setting["id"]:
            raise ValueError("Each setting needs an id")
        identity = key(setting)
        if identity in identities or setting["id"] in labels:
            raise ValueError("Duplicate source setting")
        identities.add(identity)
        labels.add(setting["id"])
        source = setting["source"]
        source_dir = absolute(source["run_dir"])
        source_path = source_dir / "run_plan.yaml"
        original = yaml.safe_load(source_path.read_text())
        original_plan = RunPlan.model_validate(original)
        pins = original_plan.pinned_step_ids()
        source_pins = {step.step_id: original_plan.probe_source_step_id(step, pins)
                       for step in original_plan.steps if step.kind in PROBE_KINDS}
        arm = setting.get("arm")
        base = repeat_base_plan(original, arm)
        rounds = repeat_rounds(arm)
        if sorted(row["repeat"] for row in setting["repeats"]) != rounds:
            raise ValueError(f"Supply every configured repeat round {rounds}, including already completed runs")
        for repeat in setting["repeats"]:
            if repeat["source"] != source:
                raise ValueError("Repeat source differs from its setting")
            args, plan_path, plan = parse_repeat(repeat)
            if (plan_path.parent in {absolute(row["source"]["run_dir"]) for row in data["settings"]}
                    or not plan_path.parent.is_relative_to(workspace)
                    or plan["run_id"] == original["run_id"]):
                raise ValueError("Each repeat must have a new run identity under the shared workspace")
            if plan["run_id"] in run_ids or plan_path.parent in run_dirs:
                raise ValueError("Duplicate repeat run identity or directory")
            run_ids.add(plan["run_id"])
            run_dirs.add(plan_path.parent)
            if args.run_id and args.run_id != plan["run_id"]:
                raise ValueError("CLI and frozen repeat run_id differ")
            if (args.persona_id, args.pa_model, args.memory_condition) != (
                    plan["persona_id"], plan["pa_model"], plan["memory_condition"]):
                raise ValueError("Runner arguments differ from the frozen setting")
            borrowed_root = absolute(str(args.borrowed_runs_root)) if args.borrowed_runs_root else plan_path.parent.parent
            if (borrowed_root / original["run_id"]).resolve() != source_dir:
                raise ValueError("Borrowed source root does not resolve to the accepted source")
            expected = copy.deepcopy(base)
            expected["run_id"] = plan["run_id"]
            original_session = original.get("pa_session_key") or f"{original['run_id']}_main"
            if plan.get("pa_session_key") != original_session:
                raise ValueError("Repeat must explicitly preserve the original PA memory session key")
            expected["pa_session_key"] = original_session
            if len(plan["steps"]) != len(expected["steps"]):
                raise ValueError("Repeat must preserve the source's complete timeline and Probe plan")
            counts = {kind: 0 for kind in PROBE_KINDS}
            for old, new in zip(expected["steps"], plan["steps"]):
                if old["kind"] in PROBE_KINDS:
                    counts[old["kind"]] += 1
                    borrow = new.get("borrow_start_from", "")
                    pin = source_pins[old["step_id"]]
                    if borrow != f"{original['run_id']}/{pin}":
                        raise ValueError("Every repeat Probe must explicitly borrow its exact original source pin")
                    metadata = read(source_dir / "checkpoints" / pin / "metadata.json", {})
                    if not metadata.get("pinned"):
                        raise ValueError(f"Missing original pinned checkpoint: {pin}")
                    old["borrow_start_from"] = borrow
            if counts != {"pre_event_probe": scope.pre_event_probes, "final_probe": scope.final_probes}:
                raise ValueError(f"A standard repeat requires all {scope.pre_event_probes} pre and "
                                 f"{scope.final_probes} final Probes")
            if expected != plan:
                raise ValueError("Repeat changed frozen scientific fields beyond run_id, original PA session and Probe source links")
            gt = plan_path.parent / "resolved_gt.yaml"
            if gt.exists() and gt.read_bytes() != (source_dir / "resolved_gt.yaml").read_bytes():
                raise ValueError("Probe repeat ground truth differs from its source")
            proof = load_repeat_source(plan_path.parent)
            if (proof is None or proof["repeat"] != repeat["repeat"]
                    or absolute(proof["source_run_dir"]) != source_dir or proof.get("arm") != arm):
                raise ValueError("Repeat needs a matching probe_repeat_source.json reference")
    return data


def probe_count() -> int:
    scope = load_run_config().probe_variance
    return scope.pre_event_probes + scope.final_probes


def production_complete(run: Path) -> bool:
    """A producer exit code does not prove that every Probe succeeded."""
    try:
        plan = yaml.safe_load((run / "run_plan.yaml").read_text())
        ledger = read(run / "ledger.json", {})
        probes = [s for s in plan["steps"] if s["kind"] in PROBE_KINDS]
        return (len(probes) == probe_count() and ledger.get("run_id") == plan["run_id"]
                and all(ledger.get("steps", {}).get(s["step_id"], {}).get("status") == "done" for s in probes))
    except (OSError, ValueError, KeyError, TypeError):
        return False


def production_settled(run: Path) -> bool:
    """Every Probe is done or failed with its attempt budget used up (N/A).

    Such a round has nothing left to produce; resuming it would only rewrite
    ledger records that its N/A scoring binds, so the worker leaves it alone.
    """
    from harness.probe_attempts import ProbeAttempts
    try:
        plan = yaml.safe_load((run / "run_plan.yaml").read_text())
        steps = read(run / "ledger.json", {}).get("steps", {})
        for probe in (s for s in plan["steps"] if s["kind"] in PROBE_KINDS):
            status = steps.get(probe["step_id"], {}).get("status")
            if status == "done":
                continue
            if status != "failed":
                return False
            try:
                ProbeAttempts(run, probe["step_id"]).next_number()
                return False
            except RuntimeError as exc:
                if "exhausted" not in str(exc) and "six total" not in str(exc):
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def scoring_handoff(directory: Path, setting: dict, repeat: dict, run: Path, score_concurrency: int) -> None:
    path = directory / f"repeat{repeat['repeat']:02d}" / "round_complete.json"
    if path.exists():
        return
    save(path, {"schema": "mempa_probe_variance_round_complete/1", "setting_id": setting["id"],
                "repeat": repeat["repeat"], "run_dir": str(run), "source": repeat["source"],
                "completed_at": now(),
                "scoring": "handoff_only_not_started", "requested_score_concurrency": score_concurrency})


async def run_setting(manifest: dict, setting: dict, options, *,
                      owner_factory=None, execute=None) -> dict:
    """One worker, one event loop, one memory owner across both run calls."""
    width = getattr(options, "rounds_per_setting", 1)
    if width not in (1, 2):
        raise ValueError("W must be 1 or 2")
    if owner_factory is None:
        from harness.shared_probe_memory import SharedProbeMemoryOwner
        owner_factory = SharedProbeMemoryOwner
    if execute is None:
        from scripts.run_orchestrator import _execute_benchmark_run
        execute = _execute_benchmark_run
    directory = setting_dir(manifest, setting)
    state_path = directory / "worker.json"
    previous = read(state_path, {})
    if previous.get("status") == "cleanup_pending":
        raise RuntimeError("Shared memory cleanup remains unresolved; no new consumers are allowed")
    state = {"setting_id": setting["id"], "pid": os.getpid(), "status": "running",
             "started_at": now(), "source": setting["source"], "rounds": previous.get("rounds", {}),
             "round_details": previous.get("round_details", {})}
    save(state_path, state)
    owner = owner_factory(directory / "shared_memory", restore_slots_root=Path(manifest["workspace_root"]) / ".probe-restore-slots",
                          restore_concurrency=options.restore_concurrency,
                          pressure_file=Path(manifest["state_root"]) / "pressure.json")
    from harness.runtime_activity import observe_run
    from scripts.run_orchestrator import GracefulStop
    stop = GracefulStop(SimpleNamespace(current_step=None), on_stop=owner.request_stop)
    originals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    stop.install()
    state["active_rounds"] = {}
    stopped = asyncio.Event()
    permits = asyncio.Semaphore(width)

    def publish():
        # Keep the W=1 fields for existing monitors; W=2 readers use the map.
        state.pop("current_repeat", None)
        state.pop("current_run_dir", None)
        if len(state["active_rounds"]) == 1:
            row = next(iter(state["active_rounds"].values()))
            state.update(current_repeat=row["repeat"], current_run_dir=row["run_dir"])
        state["updated_at"] = now()
        save(state_path, state)

    async def watch_stop():
        while not stopped.is_set():
            if (Path(manifest["state_root"]) / "stop.json").exists():
                owner.request_stop()
                return
            await asyncio.sleep(.25)

    async def run_repeat(repeat):
        async with permits:
            args, path, _ = parse_repeat(repeat)
            run = path.parent
            label = str(repeat["repeat"])
            if production_complete(run):
                state["rounds"][label] = "production_complete"
                scoring_handoff(directory, setting, repeat, run, options.score_concurrency)
                publish()
                return
            if production_settled(run):
                # N/A rounds are scored with `score_completed_run --not-applicable-exhausted`.
                state["rounds"][label] = "settled_exhausted_na"
                publish()
                return
            if owner.is_stop_requested() or (Path(manifest["state_root"]) / "stop.json").exists():
                owner.request_stop()
                return
            args.probe_concurrency = options.probe_concurrency
            state["active_rounds"][label] = {"repeat": repeat["repeat"], "run_dir": str(run)}
            state["rounds"][label] = "running"
            details = state["round_details"].setdefault(label, {"run_dir": str(run), "started_at": now()})
            publish()
            try:
                # This round's lock is released before handoff. Other rounds
                # retain their own locks and the common memory owner.
                with producer_lock(run, workspace_root=Path(manifest["workspace_root"])), observe_run(run):
                    require_retryable(run, args)
                    result = await execute(args, shared_probe_memory_owner=owner, shared_probe_stop=stop)
                if result != 0 or not production_complete(run):
                    state["rounds"][label] = "stopped" if owner.is_stop_requested() else "incomplete"
                else:
                    state["rounds"][label] = "production_complete"
                    details["finished_at"] = now()
                    scoring_handoff(directory, setting, repeat, run, options.score_concurrency)
            except BaseException as exc:
                # A failed independent round must not cancel the other round.
                # Return its error only after every task has drained.
                state["rounds"][label] = "failed"
                details["failed_at"] = now()
                state.setdefault("round_errors", {})[label] = f"{type(exc).__name__}: {exc}"
                return exc
            finally:
                state["active_rounds"].pop(label, None)
                publish()

    watcher = asyncio.create_task(watch_stop())
    group = asyncio.gather(*(run_repeat(row) for row in sorted(setting["repeats"], key=lambda row: row["repeat"])),
                           return_exceptions=True)
    try:
        try:
            outcomes = await asyncio.shield(group)
        except BaseException:
            owner.request_stop()
            # Cancellation of the worker stops new work, but never cancels
            # healthy in-flight round consumers before owner cleanup.
            while not group.done():
                try:
                    await asyncio.shield(group)
                except asyncio.CancelledError:
                    owner.request_stop()
            raise
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                raise outcome
        complete = all(state["rounds"].get(str(row["repeat"])) == "production_complete"
                       for row in setting["repeats"])
        state["status"] = ("production_complete" if complete else
                           "stopped" if owner.is_stop_requested() else "incomplete")
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stopped.set()
        try:
            await watcher
        finally:
            try:
                if not await owner.close():
                    state["status"] = "cleanup_pending"
            except BaseException as exc:
                state.update(status="cleanup_pending", cleanup_error=f"{type(exc).__name__}: {exc}")
                raise
            finally:
                for sig, handler in originals.items():
                    signal.signal(sig, handler)
                state["finished_at"] = now()
                publish()
    return state


def resources(workspace: Path, pids: set[int]) -> dict:
    """Cheap local measurements, without backend queries or model requests.

    Linux reads /proc; macOS, which has no /proc, reads `darwin_counters`.
    """
    memory, swap, processes = {}, {}, {}
    if sys.platform == "darwin":
        memory, swap, cpu, processes = darwin_counters(pids)
    else:
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                name, value = line.split(":", 1)
                if name in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                    memory[name + "_bytes"] = int(value.split()[0]) * 1024
            for line in Path("/proc/vmstat").read_text().splitlines():
                name, value = line.split()
                if name in {"pswpin", "pswpout"}:
                    swap[name] = int(value)
            cpu = [int(value) for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
        except OSError:
            cpu = []
        for pid in pids:
            try:
                rows = Path(f"/proc/{pid}/status").read_text().splitlines()
                processes[str(pid)] = {line.split(":", 1)[0] + "_bytes": int(line.split()[1]) * 1024
                                        for line in rows if line.startswith(("VmRSS:", "VmSwap:"))}
            except OSError:
                pass
    disk = shutil.disk_usage(workspace)
    return {"memory": memory, "swap_pages": swap, "cpu_ticks": cpu,
            "load_average": list(os.getloadavg()), "disk_free_bytes": disk.free,
            "disk_total_bytes": disk.total, "disk_used_bytes": disk.used, "processes": processes}


def darwin_counters(pids: set[int]) -> tuple[dict, dict, list, dict]:
    """The macOS counterparts of the /proc readings in `resources`.

    Available memory is free + speculative + inactive pages, the definition
    psutil uses on macOS. Swap pages are vm_stat's Swapins/Swapouts, process
    memory is the RSS `ps` reports, and CPU ticks are not read.
    """
    lines = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout.splitlines()
    page = int(lines[0].split("page size of", 1)[1].split()[0])
    pages = {}
    for line in lines[1:]:
        name, value = line.rsplit(":", 1)
        pages[name.strip('"')] = int(value.strip().rstrip("."))
    memory = {"MemTotal_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
              "MemAvailable_bytes": (pages["Pages free"] + pages["Pages speculative"] + pages["Pages inactive"]) * page}
    swap = {"pswpin": pages["Swapins"], "pswpout": pages["Swapouts"]}
    processes = {}
    for pid in pids:
        rss = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
        if rss:
            processes[str(pid)] = {"VmRSS_bytes": int(rss) * 1024}
    return memory, swap, [], processes


def existing_producers() -> list[dict]:
    """Inspect only known entrypoints and safe CLI fields, never environment keys.

    `ps` lists command lines the same way on Linux and macOS. It joins argv
    with spaces, so an argument that itself contains a space is split.
    """
    from harness.runtime_activity import process_start
    entrypoints = {"run_locked", "run_orchestrator", "run_probe_variance", "score_completed_run"}
    listing = subprocess.run(["ps", "-A", "-ww", "-o", "pid=,args="], capture_output=True, text=True, check=True).stdout
    rows = []
    for line in listing.splitlines():
        pid, _, command = line.strip().partition(" ")
        try:
            argv = command.split()
            entry = next((Path(token).stem if "/" in token else token.removeprefix("scripts.")
                          for token in argv
                          if (token.startswith("scripts.") and token.removeprefix("scripts.") in entrypoints)
                          or ("/" in token and Path(token).stem in entrypoints)), None)
            if entry is None:
                continue
            fields = {}
            for name in ("plan", "run-id", "run-dir", "memory-condition", "probe-concurrency", "worker-setting"):
                for index, token in enumerate(argv):
                    if token == "--" + name and index + 1 < len(argv):
                        fields[name] = argv[index + 1]
                    elif token.startswith("--" + name + "="):
                        fields[name] = token.split("=", 1)[1]
            if entry == "run_probe_variance":
                if "worker-setting" not in fields and "--execute" not in argv:
                    continue
                role = "setting_worker" if "worker-setting" in fields else "variance_dispatcher"
            else:
                role = "scorer" if entry == "score_completed_run" else "producer"
            row = {"pid": int(pid), "entrypoint": entry, "role": role, "arguments": fields,
                   "process_start": process_start(pid)}
            if "plan" in fields and Path(fields["plan"]).is_absolute():
                run = Path(fields["plan"]).parent
                ledger = read(run / "ledger.json", {})
                row["current_step"] = ledger.get("current_step")
                row["active_probes"] = sum(record.get("status") == "running"
                    for name, record in ledger.get("steps", {}).items() if name.startswith(("pre_", "fin_")))
            rows.append(row)
        except (OSError, ValueError, KeyError, IndexError):
            # A process may exit between the process listing and any read.
            continue
    return sorted(rows, key=lambda row: row["pid"])


def cgroup_limits(proc_root: Path = Path("/proc"), root: Path = Path("/sys/fs/cgroup")) -> list[dict]:
    """Report all readable cgroup-v2 ancestors that constrain this process."""
    try:
        line = next(line for line in (proc_root / "self" / "cgroup").read_text().splitlines() if line.startswith("0::"))
        current = root / line.split("::", 1)[1].lstrip("/")
    except (OSError, StopIteration):
        return []
    rows = []
    root, current = root.resolve(), current.resolve()
    while current.is_relative_to(root):
        row: dict[str, Any] = {"path": str(current)}
        for name in ("memory.max", "memory.current", "memory.swap.max", "memory.swap.current", "cpu.max"):
            try:
                raw = (current / name).read_text().strip()
                row[name] = raw if name == "cpu.max" or raw == "max" else int(raw)
            except (OSError, ValueError):
                pass
        if len(row) > 1:
            rows.append(row)
        if current == root:
            break
        current = current.parent
    return rows


def capacity_report(workspace: Path, options, *, sample_seconds: float = 0) -> dict:
    """Read-only hardware evidence and an unvalidated starting configuration.

    A hardware snapshot cannot establish per-setting restored memory usage or
    stable throughput. Those fields intentionally remain unknown until real
    formal repeats have completed under the recorded load.
    """
    workspace = Path(workspace).resolve()
    disk_root = workspace
    while not disk_root.exists():
        disk_root = disk_root.parent
    started = time.monotonic()
    first = resources(disk_root, set())
    if sample_seconds:
        time.sleep(sample_seconds)
    producers = existing_producers()
    second = resources(disk_root, {row["pid"] for row in producers})
    elapsed = time.monotonic() - started
    limits = cgroup_limits()
    memory = second["memory"]
    headrooms = [memory["MemAvailable_bytes"]] if "MemAvailable_bytes" in memory else []
    cpu_count = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    cpu_quotas = [float(cpu_count)] if cpu_count is not None else []
    for row in limits:
        limit, used = row.get("memory.max"), row.get("memory.current")
        if isinstance(limit, int) and isinstance(used, int):
            headrooms.append(max(0, limit - used))
        quota = row.get("cpu.max", "max 100000").split()
        if len(quota) == 2 and quota[0] != "max":
            cpu_quotas.append(int(quota[0]) / int(quota[1]))
    pages = None
    if sample_seconds:
        pages = {name: second["swap_pages"].get(name, 0) - first["swap_pages"].get(name, 0)
                 for name in ("pswpin", "pswpout")}
    slots = active_batch_slots(workspace / ".probe-batch-slots")
    target_slots = options.batch_slots or max(2, options.settings)
    return {"schema": "mempa_probe_variance_capacity/1", "observed_at": now(),
            "workspace_root": str(workspace), "workspace_exists": workspace.exists(),
            "disk_measurement_root": str(disk_root),
            "cpu_logical": os.cpu_count(), "cpu_affinity": cpu_count,
            "effective_cpu_quota": min(cpu_quotas) if cpu_quotas else None,
            "effective_memory_headroom_bytes": min(headrooms) if headrooms else None,
            "cgroup_limits": limits, "resources": second,
            "sample_seconds_actual": elapsed if sample_seconds else None, "swap_page_delta": pages,
            "existing_processes": producers,
            "existing_producer_count": sum(row["role"] in {"producer", "setting_worker"} for row in producers),
            "workspace_active_round_slots": slots,
            "workspace_batch_slots_current": batch_capacity(workspace / ".probe-batch-slots"),
            "candidate": {"S": options.settings, "W": options.rounds_per_setting, "P": options.probe_concurrency,
                          "R": options.restore_concurrency, "J_requested": options.score_concurrency,
                          "batch_slots": target_slots, "probe_task_upper_bound": min(options.settings * options.rounds_per_setting, target_slots) * options.probe_concurrency},
            "capacity_status": "candidate_not_validated_by_real_production",
            "highest_verified_stable_S": None, "resident_memory_per_setting_bytes": None,
            "scoring": "original_scorer_enabled_on_execute" if options.score else "handoff_only_no_score_workers",
            "execution": "not_started",
            "interpretation": ["Existing producers continue to occupy resources and the original shared slot pool.",
                               "Swap occupancy is a snapshot; only the sampled page delta shows current paging.",
                               "Eight fake workers and lock tests prove scheduling, not hardware capacity.",
                               "Confirm capacity using the remaining formal repeats; do not add synthetic repeat04/05."]}


def snapshot(manifest: dict) -> dict:
    slots = active_batch_slots(Path(manifest["workspace_root"]) / ".probe-batch-slots")
    states = []
    active_pids, probes, active_settings = set(), 0, 0
    for setting in manifest["settings"]:
        state = read(setting_dir(manifest, setting) / "worker.json", {})
        live = occupied(guard_path(manifest, setting))
        active_settings += int(live)
        if live and state.get("pid"):
            active_pids.add(state["pid"])
            active = state.get("active_rounds")
            if active is None:  # Adopt status from workers running the W=1 version.
                active = ({"legacy": {"run_dir": state["current_run_dir"]}}
                          if state.get("current_run_dir") else {})
            for row in active.values():
                ledger = read(Path(row["run_dir"]) / "ledger.json", {})
                probes += sum(step.get("status") == "running" for name, step in ledger.get("steps", {}).items()
                              if name.startswith(("pre_", "fin_")))
        states.append({"setting_id": setting["id"], "live": live, "status": state.get("status", "pending"),
                       "current_repeat": state.get("current_repeat"), "rounds": state.get("rounds", {}),
                       "active_rounds": state.get("active_rounds", {}),
                       "round_details": state.get("round_details", {}),
                       "scores": {str(row["repeat"]): read(setting_dir(manifest, setting) / f"repeat{row['repeat']:02d}" / "score.json")
                                  for row in setting.get("repeats", [])}})
    return {"updated_at": now(), "active_settings": active_settings,
            "active_rounds": sum(row.get("pid") in active_pids for row in slots), "active_probes": probes,
            "workspace_active_rounds": len(slots), "settings": states,
            "resources": resources(Path(manifest["workspace_root"]), active_pids),
            "scoring": "handoff_only_no_score_workers"}


def update_pressure(root: Path, measured: dict, options) -> dict:
    previous = read(root / "pressure.json", {})
    swap = measured["swap_pages"].get("pswpout")
    prior_swap = previous.get("swap_out_pages")
    consecutive = previous.get("consecutive_swap_out_samples", 0)
    recent = time.time() - previous.get("sample_epoch", 0) <= max(1, 3 * options.poll_seconds)
    increased = recent and prior_swap is not None and swap is not None and swap > prior_swap
    consecutive = consecutive + 1 if increased else 0
    available = measured["memory"].get("MemAvailable_bytes")
    free = measured["disk_free_bytes"]
    reasons = []
    if available is None or available < options.min_memory_gib * 1024 ** 3:
        reasons.append("memory_below_admission_threshold")
    if free < options.min_disk_gib * 1024 ** 3:
        reasons.append("disk_below_admission_threshold")
    if consecutive >= options.swap_pressure_samples:
        reasons.append("sustained_swap_out")
    value = {"paused": bool(reasons), "reasons": reasons, "sample_epoch": time.time(),
             "observed_at": now(), "swap_out_pages": swap, "consecutive_swap_out_samples": consecutive,
             "mem_available_bytes": available, "disk_free_bytes": free,
             "thresholds": {"min_memory_gib": options.min_memory_gib, "min_disk_gib": options.min_disk_gib,
                            "swap_pressure_samples": options.swap_pressure_samples},
             "action": "pause_new_settings_and_restores_keep_inflight" if reasons else "admit_new_work"}
    save(root / "pressure.json", value)
    return value


def verified_scores(run: Path) -> bool:
    from harness.judge_audit import checked_outputs
    valid, metrics_ok, errors = checked_outputs(run)
    return bool(metrics_ok and len(valid) == probe_count() and not errors)


def scoring_process_alive(state: dict) -> bool:
    from harness.runtime_activity import process_start
    return bool(state.get("pid") and state.get("process_start")
                and process_start(state["pid"]) == state["process_start"])


def score_ready_rounds(manifest: dict, options, children: dict, *, stopped: bool) -> dict:
    """Poll and dispatch only the existing scorer; never recreate a failed job."""
    from harness.runtime_activity import process_start
    for identity, (child, state_path, run) in list(children.items()):
        code = child.poll()
        if code is None:
            continue
        child.wait()
        state = read(state_path, {})
        complete = verified_scores(run)
        state.update(status="complete" if complete else "failed", exit_code=code,
                     finished_at=now(), outputs_verified=complete)
        save(state_path, state)
        children.pop(identity)
    inherited = set()
    rounds = []
    for setting in manifest["settings"]:
        for repeat in setting.get("repeats", []):
            directory = setting_dir(manifest, setting) / f"repeat{repeat['repeat']:02d}"
            handoff = read(directory / "round_complete.json")
            if handoff is None:
                continue
            state_path = directory / "score.json"
            identity = (key(setting), repeat["repeat"])
            state = read(state_path, {})
            _, plan_path, _ = parse_repeat(repeat)
            run = plan_path.parent
            if (handoff.get("repeat") != repeat["repeat"] or handoff.get("run_dir") != str(run)
                    or not production_complete(run)):
                save(state_path, {"status": "blocked", "error": "Round handoff identity or production evidence changed"})
                continue
            if identity not in children and state.get("status") in {"starting", "running"}:
                if scoring_process_alive(state):
                    inherited.add(identity)
                else:
                    complete = verified_scores(run)
                    state.update(status="complete" if complete else "interrupted", finished_at=now(),
                                 outputs_verified=complete, error=None if complete else "Lost scorer; uncertain questions are held by its request ledger")
                    save(state_path, state)
            rounds.append((identity, setting, repeat, run, state_path, state))
    active = len(children) + len(inherited)
    pending = False
    resume_dispatch = getattr(options, "scoring_dispatch_id", None)
    for identity, setting, repeat, run, state_path, state in rounds:
        may_resume = (state.get("status") == "interrupted" and resume_dispatch is not None
                      and state.get("resume_dispatch_id") != resume_dispatch)
        if (identity in children or identity in inherited or state.get("status") in {"complete", "failed", "blocked"}
                or (state.get("status") == "interrupted" and not may_resume)):
            continue
        if verified_scores(run):
            save(state_path, {"status": "complete", "outputs_verified": True, "adopted_existing_scores": True, "finished_at": now()})
            continue
        pending = True
        if stopped or (Path(manifest["state_root"]) / "stop.json").exists() or active >= options.score_concurrency:
            continue
        producer_guard = run.parent / PRODUCER_LOCKS_SEGMENT / (hashlib.sha256(str(run.resolve()).encode()).hexdigest() + ".lock")
        if occupied(producer_guard):
            continue
        argv = [sys.executable, "-m", "scripts.score_completed_run", "--run-dir", str(run), "--variance-retries"]
        prior = state
        state = {"status": "starting", "setting_id": setting["id"], "repeat": repeat["repeat"],
                 "run_dir": str(run), "source": repeat["source"],
                 "argv": argv, "started_at": now()}
        if may_resume:
            state.update(resume_dispatch_id=resume_dispatch, interrupted_resume_count=prior.get("interrupted_resume_count", 0) + 1,
                         prior_interrupted_state=prior,
                         resume_policy="original_request_ledger_holds_uncertain_questions")
        save(state_path, state)
        try:
            with (state_path.parent / "score.log").open("a") as log:
                child = subprocess.Popen(argv, cwd=ROOT,
                    env={**os.environ, "MEMPA_WORKSPACE_ROOT": manifest["workspace_root"]},
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status="running", pid=child.pid, process_start=process_start(child.pid))
            save(state_path, state)
            children[identity] = (child, state_path, run)
            active += 1
        except Exception as exc:
            state.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=now())
            save(state_path, state)
    return {"mode": "original_scorer_variance_retries", "active": active, "pending": pending,
            "limit": options.score_concurrency, "interrupted_reentry": "once_per_dispatch_request_ledger_guarded"}


def run_dispatcher(manifest_path: Path, manifest: dict, options) -> int:
    root = Path(manifest["state_root"])
    workspace = Path(manifest["workspace_root"])
    options.scoring_dispatch_id = f"{os.getpid()}-{time.monotonic_ns()}"
    with claim(workspace / ".probe-variance-dispatch.lock"), workspace_lock(workspace_root=workspace):
        save(root / "dispatch.json", {"manifest": str(manifest_path),
             "settings": options.settings, "rounds_per_setting": options.rounds_per_setting,
             "probe_concurrency": options.probe_concurrency, "restore_concurrency": options.restore_concurrency,
             "score_concurrency_requested": options.score_concurrency,
             "scoring": "original_scorer_variance_retries" if options.score else "handoff_only"})
        configure_batch_slots(workspace / ".probe-batch-slots", options.batch_slots or max(2, options.settings))

        def request_stop(signum, _frame):
            save(root / "stop.json", {"reason": "signal", "signal": signum, "requested_at": now()})

        def stop_requested():
            return (root / "stop.json").exists()

        originals = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        children: dict[str, subprocess.Popen] = {}
        score_children: dict[tuple[str, int], tuple] = {}
        attempted = set()
        try:
            while True:
                for identity, child in list(children.items()):
                    if child.poll() is not None:
                        child.wait()
                        children.pop(identity)
                        state_path = root / "settings" / identity / "worker.json"
                        state = read(state_path, {})
                        if state.get("status") in {None, "running"}:
                            state.update(status="failed", error="Worker exited without a terminal record",
                                         exit_code=child.returncode, finished_at=now())
                            save(state_path, state)
                stopped = stop_requested()
                live = len(set(children) | {key(setting) for setting in manifest["settings"]
                                           if occupied(guard_path(manifest, setting))})
                pressure = update_pressure(root, resources(workspace, set()), options)
                pending = False
                for setting in manifest["settings"]:
                    identity = key(setting)
                    state = read(setting_dir(manifest, setting) / "worker.json", {})
                    if state.get("status") == "production_complete":
                        continue
                    if occupied(guard_path(manifest, setting)) or identity in children:
                        continue
                    if state.get("status") in {"failed", "incomplete", "cleanup_pending", "running"}:
                        # Lost producers are not free retry opportunities. Their
                        # persisted attempts/namespace need an explicit recovery.
                        continue
                    if identity in attempted:
                        continue
                    pending = True
                    stopped = stopped or stop_requested()
                    if stopped or pressure["paused"] or live >= options.settings:
                        continue
                    directory = setting_dir(manifest, setting)
                    directory.mkdir(parents=True, exist_ok=True)
                    argv = [sys.executable, "-m", "scripts.run_probe_variance", "--manifest", str(manifest_path),
                            "--worker-setting", setting["id"],
                            "--rounds-per-setting", str(options.rounds_per_setting),
                            "--probe-concurrency", str(options.probe_concurrency),
                            "--restore-concurrency", str(options.restore_concurrency),
                            "--score-concurrency", str(options.score_concurrency)]
                    env = {**os.environ, "MEMPA_WORKSPACE_ROOT": str(workspace)}
                    with (directory / "worker.log").open("a") as log:
                        child = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    children[identity] = child
                    attempted.add(identity)
                    live += 1
                status = snapshot(manifest)
                scoring = (score_ready_rounds(manifest, options, score_children, stopped=stopped)
                           if options.score else {"mode": "handoff_only", "active": 0, "pending": False})
                stopped = stopped or stop_requested()
                status.update(stop_requested=stopped,
                              pressure=pressure,
                              scoring=scoring,
                              requested={"S": options.settings, "W": options.rounds_per_setting,
                                         "P": options.probe_concurrency, "R": options.restore_concurrency,
                                         "J": options.score_concurrency})
                save(root / "status.json", status)
                with (root / "activity.jsonl").open("a") as stream:
                    stream.write(json.dumps(status) + "\n")
                if (not children and not status["active_settings"] and not scoring["active"]
                        and (stopped or not (pending or scoring["pending"]))):
                    complete = all(row["status"] == "production_complete" for row in status["settings"])
                    if options.score:
                        complete = complete and all(read(setting_dir(manifest, setting) / f"repeat{row['repeat']:02d}" / "score.json", {}).get("status") == "complete"
                                                    for setting in manifest["settings"] for row in setting["repeats"])
                    return 0 if complete else 1
                time.sleep(options.poll_seconds)
        finally:
            for sig, handler in originals.items():
                signal.signal(sig, handler)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path)
    actions = result.add_mutually_exclusive_group()
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--capacity-report", action="store_true", help="Read hardware and existing producers without starting anything")
    result.add_argument("--workspace-root", type=Path, help="For a capacity report before manifests are prepared")
    result.add_argument("--sample-seconds", type=float, default=2, help="Read-only paging sample for --capacity-report, 0..10 seconds")
    result.add_argument("--min-memory-gib", type=float, default=8, help="Pause new settings/restores below this available memory")
    result.add_argument("--min-disk-gib", type=float, default=20, help="Pause new settings/restores below this workspace free space")
    result.add_argument("--swap-pressure-samples", type=int, default=3, help="Consecutive samples with swap-out growth before pausing admissions")
    result.add_argument("--settings", type=int, default=8, help="S: concurrent setting workers")
    result.add_argument("--rounds-per-setting", type=int, default=1, help="W: simultaneous rounds per setting, 1 or 2; shared memory remains owned across both")
    result.add_argument("--probe-concurrency", type=int, default=4, help="P: Probes per active round")
    result.add_argument("--restore-concurrency", type=int, default=1, help="R: restores across the workspace")
    result.add_argument("--score-concurrency", type=int, default=2, help="J: maximum original scorer jobs when --score is enabled")
    result.add_argument("--score", action="store_true", help="With --execute, run the original scorer with authorized variance retries (max J jobs)")
    result.add_argument("--batch-slots", type=int, help="Existing workspace pool capacity; default max(2, S)")
    result.add_argument("--poll-seconds", type=float, default=5)
    result.add_argument("--worker-setting", help=argparse.SUPPRESS)
    return result


def main() -> int:
    options = parser().parse_args()
    if options.rounds_per_setting not in (1, 2):
        raise ValueError("W must be 1 or 2")
    if min(options.settings, options.probe_concurrency, options.restore_concurrency, options.score_concurrency) < 1:
        raise ValueError("S/P/R/J must be positive")
    if options.probe_concurrency > 4:
        raise ValueError("P exceeds the existing validated Probe cap of 4")
    if options.batch_slots is not None and options.batch_slots < 2:
        raise ValueError("Global batch slots must remain at least 2 for existing producers; S can be 1")
    if not .1 <= options.poll_seconds <= 60:
        raise ValueError("Polling must be 0.1..60 seconds")
    if not math.isfinite(options.sample_seconds) or not 0 <= options.sample_seconds <= 10:
        raise ValueError("Hardware sampling must be 0..10 seconds")
    if (not math.isfinite(options.min_memory_gib) or options.min_memory_gib <= 0
            or not math.isfinite(options.min_disk_gib) or options.min_disk_gib <= 0
            or options.swap_pressure_samples < 1):
        raise ValueError("Resource thresholds and consecutive sample count must be positive")
    if options.capacity_report and options.worker_setting:
        raise ValueError("A capacity report cannot start a worker")
    if options.capacity_report and options.manifest is None:
        workspace = options.workspace_root or Path(os.environ.get("MEMPA_WORKSPACE_ROOT", str(ROOT / "workspace")))
        print(json.dumps(capacity_report(workspace, options, sample_seconds=options.sample_seconds), ensure_ascii=False))
        return 0
    if options.manifest is None:
        raise ValueError("--manifest is required except for a standalone --capacity-report")
    path = options.manifest.resolve()
    manifest = validate_manifest(path)
    workspace = Path(manifest["workspace_root"])
    if options.workspace_root and options.workspace_root.resolve() != workspace:
        raise ValueError("--workspace-root differs from the manifest")
    if options.capacity_report:
        print(json.dumps(capacity_report(workspace, options, sample_seconds=options.sample_seconds), ensure_ascii=False))
        return 0
    if os.environ.get("MEMPA_WORKSPACE_ROOT") and Path(os.environ["MEMPA_WORKSPACE_ROOT"]).resolve() != workspace:
        raise ValueError("Manifest workspace differs from MEMPA_WORKSPACE_ROOT")
    os.environ["MEMPA_WORKSPACE_ROOT"] = str(workspace)
    if options.worker_setting:
        setting = next(row for row in manifest["settings"] if row["id"] == options.worker_setting)
        from harness.runtime_activity import observe_model_calls
        with claim(guard_path(manifest, setting)), workspace_lock(workspace_root=workspace):
            with observe_model_calls(setting_dir(manifest, setting)):
                result = asyncio.run(run_setting(manifest, setting, options))
        return 0 if result["status"] == "production_complete" else 1
    if options.execute:
        return run_dispatcher(path, manifest, options)
    print(json.dumps({"manifest": str(path), "settings": len(manifest["settings"]),
                      "S": options.settings, "W": options.rounds_per_setting, "P": options.probe_concurrency,
                      "R": options.restore_concurrency, "J_requested": options.score_concurrency,
                      "scoring": "original_scorer_enabled_on_execute" if options.score else "handoff_only", "execution": "not_started",
                      "capacity": capacity_report(workspace, options)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
