"""Private runtime permits for three reviewed, local no-write limit paths.

Not standing tool grants: bind exact arguments/current stage, pass only on the
existing authorising turn, and recheck immediately before normal MCP dispatch.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from state_server.store import BOUNDARY_KINDS, filter_visible_blocks, stamp_visible

CONTEXT_KEY = "capability_limit_attempts"
_SPECS = {
    "shopping_order_create": ("shopping/products.json", "products",
                              {"product_id", "quantity", "delivery_address_id", "payment_method_id"}),
    "corporate_reporting_submit": ("corporate/reporting_accounts.json", "accounts", {"file_id"}),
    "utility_account_holder_update": ("finance/utility_accounts.json", "accounts", {"account_id", "remove_holder"}),
}


def matches(attempt: dict, tool: str, arguments: dict) -> bool:
    return (tool in _SPECS and tool == attempt.get("tool")
            and json.dumps(arguments, sort_keys=True) == json.dumps(attempt.get("arguments"), sort_keys=True))


def _report(root: Path, name: str, date: str | None, timestamp: str | None) -> Path | None:
    """Only a canonical, currently readable task document, never a directory."""
    path = (root / name).resolve()
    if not name.startswith("documents/") or not path.is_relative_to(root.resolve()):
        return None
    if path.relative_to(root.resolve()).as_posix() != name or not path.is_file():
        return None
    return path if filter_visible_blocks(path.read_text(), date, timestamp).strip() else None


def current(attempt: dict) -> bool:
    """Fail closed if any bound precondition changed or no longer proves a limit."""
    try:
        tool, args = attempt["tool"], attempt["arguments"]
        _, key, fields = _SPECS[tool]
        raw = Path(attempt["backing_path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != attempt["backing_sha256"] or set(args) != fields:
            return False
        strings = fields - ({"quantity"} if tool == "shopping_order_create" else set())
        if any(not isinstance(args[k], str) or not args[k] or "$" in args[k] for k in strings):
            return False
        date, timestamp = attempt.get("session_date"), attempt.get("session_timestamp")
        if tool == "shopping_order_create":
            if type(args["quantity"]) is not int or args["quantity"] < 1:
                return False
            accepts = lambda p: p.get("product_id") == args["product_id"]
        elif tool == "utility_account_holder_update":
            accepts = lambda p: p.get("account_id") == args["account_id"]
        else:
            path = _report(Path(attempt["state_root"]), args["file_id"], date, timestamp)
            if path is None or hashlib.sha256(path.read_bytes()).hexdigest() != attempt["report_sha256"]:
                return False
            accepts = lambda p: args["file_id"] in (p.get("accepted_file_ids") or [])
        records = [p for p in json.loads(raw)[key] if accepts(p)
                   and (not p.get("visible_from") or stamp_visible(str(p["visible_from"]), date, timestamp))]
        # Ambiguous selection is not one of the proven paths.
        return (len(records) == 1 and records[0].get("unavailable") is True
                and records[0].get("boundary_kind") in BOUNDARY_KINDS)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _arguments(step: dict, steps: list[dict]) -> dict | None:
    args = step.get("tool_arguments")
    if not isinstance(args, dict):
        return None
    if step["tool"] != "corporate_reporting_submit" or not str(args.get("file_id", "")).startswith("$"):
        return args.copy()
    # One proven projection: document_read of a literal canonical file returns
    # that same relative path. No searches/ranks or arbitrary runtime references.
    ref = args.get("file_id", "").split(".")
    if set(args) != {"file_id"} or len(ref) != 2 or ref[1] != "path":
        return None
    source_id = ref[0][1:]
    ids = [s.get("step_id") for s in steps]
    if len(set(ids)) != len(ids) or source_id not in ids:
        return None
    source_index, target_index = ids.index(source_id), ids.index(step["step_id"])
    source = steps[source_index]
    source_args = source.get("tool_arguments") or {}
    if (source_index >= target_index or source.get("actor") != "tool"
            or source.get("tool") != "document_read"
            or set(source_args) != {"path"} or not isinstance(source_args["path"], str)
            or "$" in source_args["path"]):
        return None
    pending, ancestors = list(step.get("depends_on") or []), set()
    while pending:
        dep = pending.pop()
        if dep in ancestors or dep not in ids:
            continue
        ancestors.add(dep)
        pending.extend(steps[ids.index(dep)].get("depends_on") or [])
    if source_id not in ancestors or any(
        s.get("actor") == "tool" and s.get("action_kind") != "read"
        for s in steps[source_index + 1:target_index]
    ):
        return None
    return {"file_id": source_args["path"]}


def prepare(draft_path: Path, state_root: Path, *, session_date: str | None = None,
            session_timestamp: str | None = None) -> list[dict[str, Any]]:
    draft = yaml.safe_load(draft_path.read_text())
    steps = (draft.get("task_execution") or {}).get("atomic_steps", [])
    result = []
    for step in steps:
        tool = step.get("tool")
        if (step.get("actor") != "tool" or tool not in _SPECS
                or step.get("expects_capability_limit") is not True):
            continue
        args = _arguments(step, steps)
        path = state_root / _SPECS[tool][0]
        if args is None or not path.is_file():
            continue
        attempt = dict(step_id=step["step_id"], tool=tool, arguments=args,
                       declared_arguments=step.get("tool_arguments"),
                       backing_path=str(path.resolve()), backing_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                       state_root=str(state_root.resolve()), session_date=session_date,
                       session_timestamp=session_timestamp)
        if tool == "corporate_reporting_submit":
            try:
                report = _report(state_root, args.get("file_id", ""), session_date, session_timestamp)
                if report is None:
                    continue
                attempt["report_sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
            except (OSError, ValueError, TypeError):
                continue
        if current(attempt):
            result.append(attempt)
    return result
