"""OpenRouter upstreams a run stops sending one model's requests to.

OpenRouter serves one model id from many upstream providers and names the one
that answered in each response (`upstream_provider=` on the `openrouter_call`
line). One upstream can fail the same way over and over --
DigitalOcean answered `deepseek/deepseek-v4-flash` File memory consolidations
with empty tool calls until the run stopped six times at one session -- and
OpenRouter keeps routing to it. So once one model has failed `threshold`
times on one upstream within a run, every later request for that model carries
that upstream in OpenRouter's `provider.ignore` (`OpenAICompatProvider`).
A failure counted is one the upstream is to blame for: an error that names it,
or an empty reply -- no non-blank text and no tool call carrying arguments
(`{}`, blank or missing arguments carry none) -- cut off at the token limit or
rejected by its caller (a PA or File memory consolidation reply that breaks its
contract). A rejected reply with text or tool arguments is the model's own
failure, the same on every upstream, and is not counted. When OpenRouter
answers that the list leaves the model no upstream, the model's counts and
ignored upstreams are cleared (`reset`) and the request is sent once more
without the list.

`<run_dir>/provider_avoidance.json` holds, per model, the failures counted on
each upstream and the upstreams ignored, each with when and on which failure:

    {"schema": "mempa_provider_avoidance/1",
     "models": {"deepseek/deepseek-v4-flash": {
        "failures": {"DigitalOcean": 3, "Baidu": 1},
        "ignored": {"DigitalOcean": {"at": "<UTC ISO time>",
                                     "failure_kind": "contract_rejected",
                                     "failure": "file_consolidation: Missing save_memory fields: ..."}}}}}

The PA, File memory consolidation, Mem0 extraction (on its own thread) and
every probe branch's PA of one run read and count into this one file, and
concurrent probes count at the same time, so each update re-reads the file
under an exclusive lock beside it (`provider_avoidance.lock`) and replaces it
atomically. A resume loads it; a file that exists but is invalid stops the
launch, and a missing file is a fresh start with nothing counted or ignored.
A plan frozen without `provider_avoid_after_failures` runs without this state.
"""

from __future__ import annotations

import fcntl
import json
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from harness.shared_probe_memory import _atomic

STATE_FILENAME = "provider_avoidance.json"
SCHEMA = "mempa_provider_avoidance/1"
#: What counts against an upstream: an HTTP or provider error that names it
#: (`provider_error`), a `length` response with no content and no tool
#: arguments (`provider_truncation`), and a response with no content and no
#: tool arguments that its caller rejected under its own contract
#: (`contract_rejected`). A rejected response with content or tool arguments is
#: the model's failure, not the upstream's, and is not counted.
COUNTED_FAILURE_KINDS = ("provider_error", "provider_truncation", "contract_rejected")
#: The stored failure text is cut to this many characters.
_FAILURE_TEXT_LIMIT = 300
_MODEL_KEYS = {"failures", "ignored"}
_IGNORED_KEYS = {"at", "failure_kind", "failure"}


class ProviderAvoidance:
    """One run's avoidance state; `threshold` 0 turns it off (nothing read, counted or ignored)."""

    def __init__(self, run_dir: Path | str, *, threshold: int) -> None:
        if type(threshold) is not int or threshold < 0:
            raise ValueError(f"provider avoidance threshold must be a non-negative integer: value={threshold!r}")
        self.path = Path(run_dir) / STATE_FILENAME
        self.threshold = threshold
        if self.enabled:
            # A resume loads the state here, so an invalid file stops the
            # launch rather than the first request.
            self._read()

    @property
    def enabled(self) -> bool:
        return self.threshold > 0

    def ignored_providers(self, model: str) -> list[str]:
        """The upstreams `model`'s requests must skip, read fresh from the file."""
        if not self.enabled:
            return []
        return sorted(self._read()["models"].get(model, {}).get("ignored", {}))

    def record_failure(self, model: str, provider: str, *, kind: str, failure: str) -> None:
        """Count one failure of `model` on `provider`; ignore the provider at the threshold."""
        if not self.enabled:
            return
        if kind not in COUNTED_FAILURE_KINDS:
            raise ValueError(f"unknown provider failure kind: value={kind!r}")
        with self._locked():
            state = self._read()
            entry = state["models"].setdefault(model, {"failures": {}, "ignored": {}})
            count = entry["failures"].get(provider, 0) + 1
            entry["failures"][provider] = count
            newly_ignored = count >= self.threshold and provider not in entry["ignored"]
            if newly_ignored:
                entry["ignored"][provider] = {
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "failure_kind": kind,
                    "failure": failure[:_FAILURE_TEXT_LIMIT],
                }
            _atomic(self.path, state)
        if newly_ignored:
            print(f"provider_avoided model={model} provider={provider} failures={count}",
                  file=sys.stderr, flush=True)

    def reset(self, model: str) -> None:
        """Forget `model`'s failure counts and ignored upstreams.

        For OpenRouter's answer that the ignore list left the model no upstream:
        the list is kept across resumes, so without this every later request
        for the model would be refused the same way.
        """
        with self._locked():
            state = self._read()
            entry = state["models"].pop(model, {"ignored": {}})
            _atomic(self.path, state)
        print(f"provider_avoidance_reset model={model} cleared={','.join(sorted(entry['ignored']))}",
              file=sys.stderr, flush=True)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        with self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _read(self) -> dict[str, Any]:
        """The state on disk, validated; an empty state when the file does not exist yet."""
        if not self.path.exists():
            return {"schema": SCHEMA, "models": {}}

        def invalid(detail: str) -> ValueError:
            return ValueError(f"provider avoidance state {self.path} is invalid: {detail}")

        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise invalid(f"not JSON ({exc})") from exc
        if not isinstance(state, dict) or set(state) != {"schema", "models"} or state["schema"] != SCHEMA:
            raise invalid(f"expected the keys schema ({SCHEMA}) and models")
        if not isinstance(state["models"], dict):
            raise invalid("models is not a mapping")
        for model, entry in state["models"].items():
            if not isinstance(entry, dict) or set(entry) != _MODEL_KEYS:
                raise invalid(f"model {model!r} has the keys {sorted(entry) if isinstance(entry, dict) else entry!r}")
            failures, ignored = entry["failures"], entry["ignored"]
            if not isinstance(failures, dict) or not all(
                    isinstance(provider, str) and provider and type(count) is int and count >= 1
                    for provider, count in failures.items()):
                raise invalid(f"model {model!r} failures is not a mapping of provider to a positive count: {failures!r}")
            if not isinstance(ignored, dict):
                raise invalid(f"model {model!r} ignored is not a mapping")
            for provider, record in ignored.items():
                if provider not in failures:
                    raise invalid(f"model {model!r} ignores {provider!r} with no failure counted")
                if (not isinstance(record, dict) or set(record) != _IGNORED_KEYS
                        or not isinstance(record["at"], str) or not record["at"]
                        or record["failure_kind"] not in COUNTED_FAILURE_KINDS
                        or not isinstance(record["failure"], str)):
                    raise invalid(f"model {model!r} ignored record for {provider!r} is malformed: {record!r}")
        return state
