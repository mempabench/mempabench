"""Read-only production measurements; never dispatch, score, or calculate variance."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

from harness.run_config import load_run_config
from harness.runtime_activity import process_start


def read(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def timestamp(value):
    return datetime.fromisoformat(value).timestamp()


def retrieval_summary(run: Path) -> dict:
    path = run / "retrieval_activity.jsonl"
    durations = []
    failures = 0
    if path.exists():
        # Concurrent append may leave the final line temporarily incomplete.
        for line in path.read_text().splitlines(keepends=True):
            if not line.endswith("\n"):
                continue
            row = json.loads(line)
            if row.get("boundary") != "native_logical_search" or row.get("run_dir") != str(run):
                continue
            durations.append(row["elapsed_ms"] / 1000)
            failures += not row["success"]
    return {"boundary": "native_logical_search", "samples": len(durations), "failures": failures,
        "p95_seconds": sorted(durations)[math.ceil(.95 * len(durations)) - 1] if durations else None,
        "percentile_method": "nearest_rank", "includes_failed_queries": True}


def sample(manifest: dict, round_starts: dict | None = None) -> dict:
    if round_starts is None:
        round_starts = {}
    root = Path(manifest["state_root"])
    status = read(root / "status.json", {})
    # How many probes a complete repeat runs: `probe_variance` in configs/run_config.yaml.
    scope = load_run_config().probe_variance
    expected = {f"{kind}_{number:02d}" for kind, total in
                (("pre", scope.pre_event_probes), ("fin", scope.final_probes))
                for number in range(1, total + 1)}
    rows, shared, requests = [], [], []
    for setting in manifest["settings"]:
        identity = hashlib.sha256(str(Path(setting["source"]["run_dir"]).resolve()).encode()).hexdigest()
        directory = root / "settings" / identity
        worker = read(directory / "worker.json", {})
        pid = worker.get("pid")
        live = bool(pid and process_start(pid))
        setting_activity = read(directory / "runtime_activity.json")
        if (setting_activity and setting_activity.get("state") == "running" and
                process_start(setting_activity.get("pid", 0)) == setting_activity.get("process_start")):
            requests.extend(setting_activity.get("requests", []))
        for path in sorted((directory / "shared_memory").glob("*.json")):
            record = read(path)
            shared.append({"setting_id": setting["id"], "path": str(path),
                "checkpoint_id": record.get("checkpoint_id"), "backend": record.get("memory_condition"),
                "state": record.get("state"), "restore_count": record.get("restore_count"),
                "prepare_seconds": record.get("prepare_seconds"),
                "consumers": dict(Counter(record.get("consumers", {}).values()))})
        for repeat in setting["repeats"]:
            argv = repeat["argv"]
            run = Path(argv[argv.index("--plan") + 1]).parent
            ledger = read(run / "ledger.json", {})
            steps = {key: value for key, value in ledger.get("steps", {}).items()
                     if key.startswith(("pre_", "fin_"))}
            counts = Counter(value.get("status") for value in steps.values())
            attempts = {}
            for path in sorted((run / "probe_attempts").glob("*.json")):
                record = read(path)
                used = record.get("attempts", [])
                attempts[record["step_id"]] = {"used": len(used),
                    "statuses": [attempt.get("status") for attempt in used]}
            complete = set(steps) == expected and counts["done"] == len(expected)
            starts = [timestamp(value["started_at"]) for value in steps.values() if value.get("started_at")]
            ends = [timestamp(value["ended_at"]) for value in steps.values() if value.get("ended_at")]
            details = worker.get("round_details", {}).get(str(repeat["repeat"]), {})
            started = details.get("started_at")
            if started is None and worker.get("current_repeat") == repeat["repeat"]:
                started = worker.get("updated_at")
            if started:
                round_starts.setdefault(str(run), {"started_at": started, "evidence": str(directory / "worker.json")})
            start_proof = round_starts.get(str(run))
            wall = (max(ends) - timestamp(start_proof["started_at"])
                    if complete and len(ends) == len(expected) and start_proof else None)
            if complete and start_proof and details.get("finished_at"):
                wall = timestamp(details["finished_at"]) - timestamp(start_proof["started_at"])
            activity = read(run / "runtime_activity.json")
            if activity is None and worker.get("current_repeat") == repeat["repeat"]:
                activity = setting_activity
            active_calls = (activity.get("requests", []) if activity and activity.get("state") == "running" and
                            process_start(activity.get("pid", 0)) == activity.get("process_start") else None)
            rows.append({"setting_id": setting["id"], "repeat": repeat["repeat"],
                "run_dir": str(run), "worker_live": live, "worker_pid": pid,
                "worker_process_start": process_start(pid) if pid else None,
                "production_complete": complete, "probe_counts": dict(counts),
                "expected_probes": len(expected),
                "attempts": attempts,
                "retried_probes": [probe for probe, budget in attempts.items() if budget["used"] > 1],
                "exhausted_probes": [probe for probe, budget in attempts.items()
                    if budget["used"] >= 3 and budget["statuses"][-1] not in {"started", "succeeded"}
                    and steps.get(probe, {}).get("status") != "done"],
                "production_start": start_proof,
                "first_probe_started_epoch": min(starts) if starts else None,
                "last_probe_ended_epoch": max(ends) if complete and ends else None,
                "production_wall_seconds": wall,
                "probes_per_hour": len(expected) * 3600 / wall if wall and wall > 0 else None,
                "inflight_model_requests": len(active_calls) if active_calls is not None else None,
                "retrieval": retrieval_summary(run),
                "score": read(directory / f"repeat{repeat['repeat']:02d}" / "score.json", {}).get("status")})
    return {"observed_at": datetime.now(timezone.utc).isoformat(),
        "dispatcher_observed_at": status.get("updated_at"),
        "active_settings": status.get("active_settings"), "active_rounds": status.get("active_rounds"),
        "active_probes": status.get("active_probes"), "configured": status.get("requested"),
        "resources": status.get("resources"), "pressure": status.get("pressure"),
        "shared_states": dict(Counter(row["state"] for row in shared)),
        "resident_instances": sum(row["state"] == "ready" for row in shared),
        "shared_instances": shared, "rounds": rows,
        "observed_model_roles": dict(Counter(row["role"] for row in requests)),
        "measurement_limits": ["Missing runtime_activity means unknown request count, not zero.",
            "Production wall requires observed execution start; first Probe time is not substituted.",
            "Historical production did not persist retrieval durations; query p95 is unavailable.",
            "Ready manifests count resident memory states; preparing/retained states are separate.",
            "Overlapping active rounds establish observed concurrency, not validated stable capacity."]}


def summarize(samples: list[dict], last: dict) -> dict:
    groups = {}
    for row in samples:
        g = row["active_rounds"]
        group = groups.setdefault(str(g), {"active_rounds": g, "samples": 0,
            "min_available_memory_bytes": None, "max_resident_instances": 0})
        group["samples"] += 1
        available = (row.get("resources") or {}).get("memory", {}).get("MemAvailable_bytes")
        if available is not None:
            prior = group["min_available_memory_bytes"]
            group["min_available_memory_bytes"] = available if prior is None else min(available, prior)
        group["max_resident_instances"] = max(group["max_resident_instances"], row["resident_instances"])
    return {"schema": "mempa_probe_variance_measurements/1", "updated_at": last["observed_at"],
        "measurement_started_at": samples[0]["observed_at"],
        "highest_observed_G": max(row["active_rounds"] or 0 for row in samples),
        "highest_complete_validated_G": None,
        "capacity_validation": "Requires complete production, scoring, isolation and cleanup evidence; not inferred from CPU count.",
        "concurrency_measurements": list(groups.values()),
        "completed_rounds": sum(row["production_complete"] for row in last["rounds"]),
        "last_sample": last}


def write_table(path: Path, report: dict) -> None:
    lines = ["# A/C Probe 并发测量", "", f"采样更新：{report['updated_at']}", "",
        "活跃轮数 G、setting 数与驻留实例分别计数。完整验证并发尚未确认。", "",
        "| 实际 G | 样本数 | 最低可用内存 GiB | 最多 ready 实例 |",
        "|---:|---:|---:|---:|"]
    for row in report["concurrency_measurements"]:
        available = row["min_available_memory_bytes"]
        memory = f"{available / 1024 ** 3:.2f}" if available is not None else "未知"
        lines.append(f"| {row['active_rounds']} | {row['samples']} | {memory} | {row['max_resident_instances']} |")
    lines += ["", "| Setting | Repeat | 完成 Probe | 完整生产秒数 | Probe/小时 | 查询 p95 秒 |",
        "|---|---:|---:|---:|---:|---:|"]
    def number(value):
        return "未齐备" if value is None else f"{value:.3f}"
    for row in report["last_sample"]["rounds"]:
        lines.append(f"| {row['setting_id']} | {row['repeat']} | "
                     f"{row['probe_counts'].get('done', 0)}/{row['expected_probes']} | "
                     f"{number(row['production_wall_seconds'])} | {number(row['probes_per_hour'])} | "
                     f"{number(row['retrieval']['p95_seconds'])} |")
    lines += ["", "生产墙钟包含准备、恢复与暂停，不含评分；未完成或缺起始证据时不填推算值。",
        "查询 p95 使用记录到的完整逻辑检索耗时（含失败），旧版本未记录的检索保留缺口。",
        "本表不计算任何科学指标或方差；原始证据见同目录 samples.jsonl 和 summary.json。", ""]
    temporary = path.with_suffix(".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--watch-pid", type=int)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args(argv)
    if args.interval < 1:
        parser.error("interval must be at least one second")
    manifest = read(args.manifest)
    expected_start = process_start(args.watch_pid) if args.watch_pid else None
    if args.watch_pid and expected_start is None:
        parser.error("watched process is not live; do not infer or restart it")
    args.output.mkdir(parents=True, exist_ok=True)
    samples = []
    round_starts = read(args.output / "round_starts.json", {})
    while True:
        row = sample(manifest, round_starts)
        atomic(args.output / "round_starts.json", round_starts)
        # Keep history small; per-round details remain in the append-only log.
        samples.append({name: row[name] for name in
                        ("observed_at", "active_rounds", "resources", "resident_instances")})
        with (args.output / "samples.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report = summarize(samples, row)
        atomic(args.output / "summary.json", report)
        write_table(args.output / "concurrency.md", report)
        if not args.watch_pid or process_start(args.watch_pid) != expected_start:
            break
        time.sleep(args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
