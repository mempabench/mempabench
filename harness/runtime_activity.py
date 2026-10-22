"""Observe managed model calls without storing prompts, replies, or credentials.

The managed entrypoints install these hooks for their process lifetime. They do
not change call arguments, return values, retries, or stream consumption.
"""
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import importlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import uuid
import warnings

_run_activity = ContextVar("probe_run_activity", default=None)
_search_backends = ContextVar("observed_search_backends", default=frozenset())


def process_start(pid):
    """Start time of a live process (`ps -o lstart`, in UTC), or None once it is gone.

    With the PID it identifies one process, so a reused PID is not mistaken for
    the original. `ps` reports it the same way on Linux and macOS.
    """
    found = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True,
                           env={**os.environ, "LC_ALL": "C", "TZ": "UTC"})
    return found.stdout.strip() or None


class Activity:
    def __init__(self, run, *, filename="runtime_activity.json"):
        self.path = Path(run) / filename
        self.run_dir = str(Path(run).resolve())
        self.active = {}
        self.lock = threading.RLock()
        self.pid = os.getpid()
        self.start = process_start(self.pid)
        self.warned = False
        self.state = "running"
        self._retrieval = None

    @property
    def retrieval(self):
        with self.lock:
            if self._retrieval is None:
                self._retrieval = Activity(self.path.parent, filename="retrieval_activity.json")
                self._retrieval.state = self.state
            return self._retrieval

    def finish(self):
        with self.lock:
            self.publish("draining" if self.active else "stopped")
            if self._retrieval is not None:
                self._retrieval.finish()

    def publish(self, state=None):
        if state is not None:
            self.state = state
        if not self.path.parent.is_dir():
            return
        row = {"pid": self.pid, "process_start": self.start, "state": self.state,
               "updated_at": datetime.now(timezone.utc).isoformat(),
               "requests": list(self.active.values())}
        try:
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(row) + "\n")
            temporary.replace(self.path)
        except OSError as exc:
            if not self.warned:
                warnings.warn(f"Cannot write runtime activity: {type(exc).__name__}")
                self.warned = True

    @contextmanager
    def _request(self, role, run_dir=None):
        token = uuid.uuid4().hex
        with self.lock:
            if self.state == "stopped":
                self.state = "draining"
            self.active[token] = {"role": role, "started_at": datetime.now(timezone.utc).isoformat()}
            if run_dir is not None:
                self.active[token]["run_dir"] = run_dir
            self.publish()
        try:
            yield
        finally:
            with self.lock:
                self.active.pop(token)
                if not self.active and self.state == "draining":
                    self.state = "stopped"
                self.publish()

    @contextmanager
    def request(self, role):
        scoped = _run_activity.get()
        with ExitStack() as stack:
            stack.enter_context(self._request(role, scoped.run_dir if scoped is not None else None))
            if scoped is not None and scoped is not self:
                stack.enter_context(scoped._request(role))
            yield

    def wrap(self, method, role):
        if inspect.isasyncgenfunction(method):
            @wraps(method)
            async def observed(*args, **kwargs):
                with self.request(role):
                    stream = method(*args, **kwargs)
                    try:
                        async for item in stream:
                            yield item
                    finally:
                        await stream.aclose()
        elif inspect.iscoroutinefunction(method):
            @wraps(method)
            async def observed(*args, **kwargs):
                with self.request(role):
                    return await method(*args, **kwargs)
        else:
            @wraps(method)
            def observed(*args, **kwargs):
                with self.request(role):
                    return method(*args, **kwargs)
        return observed

    def wrap_search(self, method, backend):
        """One native logical search, including embedding and internal retries."""
        @wraps(method)
        def observed(*args, **kwargs):
            active = _search_backends.get()
            if backend in active:
                return method(*args, **kwargs)
            token = _search_backends.set(active | {backend})
            try:
                return measured(*args, **kwargs)
            finally:
                _search_backends.reset(token)

        def measured(*args, **kwargs):
            scoped = _run_activity.get()
            sinks = [self.retrieval]
            if scoped is not None and scoped is not self:
                sinks.append(scoped.retrieval)
            run_dir = scoped.run_dir if scoped is not None else self.run_dir
            with ExitStack() as stack:
                for sink in sinks:
                    stack.enter_context(sink._request(backend, run_dir))
                started_at = datetime.now(timezone.utc).isoformat()
                started = time.monotonic()
                error = None
                try:
                    return method(*args, **kwargs)
                except BaseException as exc:
                    error = type(exc).__name__
                    raise
                finally:
                    row = {"schema": "mempa_retrieval_activity/1", "boundary": "native_logical_search",
                           "run_dir": run_dir, "backend": backend, "started_at": started_at,
                           "finished_at": datetime.now(timezone.utc).isoformat(),
                           "elapsed_ms": (time.monotonic() - started) * 1000,
                           "success": error is None, "error_type": error}
                    for sink in sinks:
                        with sink.lock:
                            try:
                                with (sink.path.parent / "retrieval_activity.jsonl").open("a") as stream:
                                    stream.write(json.dumps(row) + "\n")
                            except OSError as exc:
                                if not sink.warned:
                                    warnings.warn(f"Cannot write retrieval activity: {type(exc).__name__}")
                                    sink.warned = True
        return observed


@contextmanager
def observe_run(run):
    """Attribute this task and its to_thread calls without installing new hooks."""
    activity = Activity(run)
    token = _run_activity.set(activity)
    activity.publish()
    try:
        yield
    finally:
        _run_activity.reset(token)
        activity.finish()


@contextmanager
def observe_model_calls(run):
    activity = Activity(run)
    targets = [
        ("pa.providers.openai_compat_provider", "OpenAICompatProvider", "chat", "pa_or_memory_lm"),
        ("pa.providers.openai_compat_provider", "OpenAICompatProvider", "chat_stream", "pa_or_memory_lm"),
        ("pa.providers.anthropic_provider", "AnthropicProvider", "chat", "pa_or_memory_lm"),
        ("pa.providers.anthropic_provider", "AnthropicProvider", "chat_stream", "pa_or_memory_lm"),
        ("simulator.providers.openai_compat_provider", "OpenAICompatProvider", "chat", "simulator"),
        ("simulator.providers.anthropic_provider", "AnthropicProvider", "chat", "simulator"),
        ("simulator.judge.client", "JudgeClient", "generate", "runtime_judge"),
        ("judge.evaluator", "OpenRouterJudgeClient", "complete_payload", "final_probe_judge"),
    ]
    originals = []
    try:
        for module, name, method, role in targets:
            cls = getattr(importlib.import_module(module), name)
            original = getattr(cls, method)
            originals.append((cls, method, original))
            setattr(cls, method, activity.wrap(original, role))
        # The read-only override includes post-search integrity checks. Its
        # super call is suppressed by the context guard, so it counts once.
        for module, name, method, backend in [
            ("memory.backends.mem0_client", "_ValidatedMem0Client", "search", "mem0_or_simple_rag"),
            ("memory.backends.mem0_read_only", "ReadOnlyMem0Client", "search", "mem0_or_simple_rag"),
            ("memory.backends.memos_client", "MemOSHttpClient", "search_memories", "memos"),
        ]:
            cls = getattr(importlib.import_module(module), name)
            original = getattr(cls, method)
            originals.append((cls, method, original))
            setattr(cls, method, activity.wrap_search(original, backend))
        activity.publish()
        yield
    finally:
        for cls, method, original in reversed(originals):
            setattr(cls, method, original)
        activity.finish()
