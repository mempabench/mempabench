"""Verification-only override of the proactive outreach branch.

Owner-approved 2026-09-08, for one purpose: `acc_007` (`proactive_outreach`,
triggered two-beat session) has run nine rounds and the PA chose `send` every
time, so the draft's `opening_after_suppress` half has never been exercised
live. `MEMPA_VERIFY_OUTREACH=suppress` forces that branch once.

**A run under this variable is not a scored run.** Three things say so, and
none of them can be turned off independently of the override itself:

* the model's own selection is recorded in the IX evidence *first* and left
  alone -- the override is a second record beside it, never a rewrite of it;
* the step's transcript carries one `verification_override` event, and its
  `meta.yaml` the matching `verification_override` block;
* `experiment_admin inspect` reports that block as a finding.

Unset is the only value that touches nothing: `verify_outreach_action` returns
None and every caller short-circuits, so an ordinary run runs the code it ran
yesterday. Any value other than `send` / `suppress` raises at run start rather
than being read as a default -- a knob that invalidates its run has no partial
interpretation.
"""

from __future__ import annotations

import os

#: The one variable. Named `MEMPA_VERIFY_` rather than `MEMPA_` so it does not
#: read like the per-step env the state runtime exports (`MEMPA_GATE` etc).
ENV_VAR = "MEMPA_VERIFY_OUTREACH"

#: The `ix_event` name the wrapper appends, and the transcript / meta key the
#: harness writes from it. One string, so a reader grepping for it finds every
#: plane the stamp reaches.
VERIFICATION_OVERRIDE = "verification_override"

#: `proactive_outreach` has exactly two settings, one per action (§3.1 that
#: row). Forcing the action without the setting would leave the renderer's own
#: `High -> send` / `Low -> suppress` rule to undo the override.
OVERRIDE_SETTINGS = {"send": "High", "suppress": "Low"}


def verify_outreach_action() -> str | None:
    """`send` / `suppress` when the variable is set, None when it is not."""
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return None
    if raw not in OVERRIDE_SETTINGS:
        raise RuntimeError(
            f"{ENV_VAR} must be 'send' or 'suppress' (this is a verification "
            f"run, not a scored one); got {raw!r}"
        )
    return raw
