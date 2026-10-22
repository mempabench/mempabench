"""Force one IX setting into a live PA, for validation runs only.

Some validation runs need a session whose
`process_visibility` / `autonomy_level` is a setting the persona matrix never
produces. The simulator's preference overlay only rewrites the *user* matrix, so
nothing of it reaches the PA. This seeds the PA's own per-session selection cache
instead: the setting is then inherited like any first-turn choice, and the
attribute's `IX_` selection tool is no longer offered for that session.

`proactive_outreach` is the one attribute whose seed reaches the model turn by
a different route. The environment-event turn inherits no settings by design
(`AgentLoop._run_agent_loop`), so `IX_proactive_outreach` stays on the turn --
it is the only carrier of `outreach_message`, and dropping it would leave the
PA no way to write the candidate contact at all. The seed instead survives as
`IXSelection.forced`, which `_remember_ix_selections` refuses to overwrite, so
the renderer reads the forced setting and applies §3.1's override (`Low` ->
suppress with the message held, `High` -> send).
"""

from __future__ import annotations

from typing import Any

from pa.agent.ix.events import IXSelection
from pa.agent.ix.specs import IX_PASSIVE_SPECS

FORCED_EVIDENCE = "forced by --force-ix (validation run)"


def seed_ix_selection(
    pa: Any,
    *,
    session_key: str,
    script_session_id: str | None,
    attribute: str,
    setting: str,
) -> IXSelection:
    """Pre-select `setting` for `attribute` in one benchmark session."""
    spec = IX_PASSIVE_SPECS[attribute]  # unknown attribute: fail fast
    if setting not in spec.settings:
        raise ValueError(
            f"unknown setting for {attribute}: {setting!r} "
            f"(known: {', '.join(spec.settings)})"
        )
    selection = IXSelection(
        attribute=attribute,
        tool_name=spec.name,
        setting=setting,
        evidence=FORCED_EVIDENCE,
        application="",
        forced=True,
    )
    cache_key = (session_key, str(script_session_id or ""))
    pa._ix_selection_cache.setdefault(cache_key, {})[attribute] = selection
    return selection
