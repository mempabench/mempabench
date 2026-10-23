"""Durable whole-Probe attempt intents for the shared-memory repeat runner."""
from __future__ import annotations
import json
import fcntl
from contextlib import contextmanager
from pathlib import Path
from harness.shared_probe_memory import _atomic


class ProbeAttempts:
    def __init__(self, run_dir: Path, step_id: str):
        self.path = Path(run_dir) / "probe_attempts" / f"{step_id}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.value = json.loads(self.path.read_text()) if self.path.exists() else {"step_id": step_id, "attempts": []}
        self._active_number = None

    @contextmanager
    def _locked(self):
        with self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                self.value = json.loads(self.path.read_text())
            yield

    def next_number(self) -> int:
        with self._locked():
            return self._next_number()

    def _next_number(self) -> int:
        attempts = self.value["attempts"]
        if attempts and attempts[-1]["status"] == "started":
            raise RuntimeError("Prior Probe attempt has unknown outcome; reconcile its paid requests before replay")
        if attempts and attempts[-1]["status"] == "succeeded":
            raise RuntimeError("A successful Probe must not be regenerated")
        from harness.probe_na import ATTEMPT_LIMIT, TOTAL_ATTEMPT_LIMIT, counted_failures
        if counted_failures(attempts[self.budget_start():]) >= ATTEMPT_LIMIT:
            raise RuntimeError("Probe exhausted its three counted attempts")
        if len(attempts) >= TOTAL_ATTEMPT_LIMIT:
            raise RuntimeError("Probe used its six total attempts, including network failures")
        return len(attempts) + 1

    def budget_start(self) -> int:
        """Attempts before a user-authorized reopen do not count toward the three."""
        return int(self.value.get("reopened", {}).get("after_attempt", 0))

    def ceiling(self) -> int:
        from harness.probe_na import ATTEMPT_LIMIT
        return self.budget_start() + ATTEMPT_LIMIT

    def reopen(self, authorization: str) -> None:
        """Give an exhausted Probe one more budget of three attempts (six in total at most).

        Only by an explicit user decision;
        the earlier failed attempts stay recorded as they were.
        """
        from harness.probe_na import ATTEMPT_LIMIT, TOTAL_ATTEMPT_LIMIT, counted_failures
        with self._locked():
            attempts = self.value["attempts"]
            if ("reopened" in self.value or not attempts or any(a["status"] != "failed" for a in attempts)
                    or counted_failures(attempts) < ATTEMPT_LIMIT or len(attempts) >= TOTAL_ATTEMPT_LIMIT):
                raise RuntimeError("Only an exhausted, never reopened Probe with total attempts left may be reopened")
            self.value["reopened"] = {"after_attempt": len(attempts), "authorization": authorization}
            _atomic(self.path, self.value)

    def begin(self) -> int:
        with self._locked():
            number = self._next_number()
            self.value["attempts"].append({"attempt": number, "status": "started"})
            _atomic(self.path, self.value)
            self._active_number = number
            return number

    def finish(self, *, error: BaseException | None = None) -> None:
        with self._locked():
            attempts = self.value["attempts"]
            if (self._active_number is None or not attempts
                    or attempts[-1]["attempt"] != self._active_number
                    or attempts[-1]["status"] != "started"):
                raise RuntimeError("Only the owner of a started attempt may finish it")
            current = attempts[-1]
            current["status"] = "failed" if error is not None else "succeeded"
            if error is not None:
                current["error"] = f"{type(error).__name__}: {error}"
            _atomic(self.path, self.value)
            self._active_number = None
