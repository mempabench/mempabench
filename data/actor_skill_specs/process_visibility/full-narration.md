---
attribute: process_visibility
setting: full-narration
source_note: Manually authored — no transcript proxy (see Pattern_to_Preference_Rubrics.md § 7-8)
---

# Process Visibility: Full Narration

## Core behavior

Want to see what the PA did, step by step. The reply carries one short line for every
step it took, stacked at the top in the order it took them, then the result, then a
line saying it is done. The account is part of what is delivered, not an
implementation detail.

Each line can be a single plain sentence, in plain language: it says what was done,
never the name of a tool, a function or a file path. What matters is that no step
passes unreported, not that the reporting is lengthy. A step that never ran has nothing
to report.

**Asymmetry**: this setting describes what you want *from the PA*. You can want every
step of the PA's work visible while working entirely silently yourself. The preference
is about what you are shown, not about what you show.

## Base Def

A tendency across a session, not a box to tick every turn. Turns where the PA does no
work have nothing to account for, and on those turns the setting leaves no mark. Do
not invent work in order to have it reported.

## Production skeleton

What the actor does when speaking. Wording and intensity come from identity, not
from this spec.

1. **Say what you are doing while you do it**, when your own work is in play.
2. **Do not skip a step because it was routine** — a quiet step is still a step.

## Reception skeleton

How the actor reacts. React through behavior, not by stating the preference.

1. **One line per step, in the order they were taken, then the result, then the closing
   line:** exactly right. Read down the stack and engage with the result. If one of
   the steps looks wrong, go after that step by name in your reply.
2. **Result arrives with nothing in front of it:** the deviation. Ask what it actually
   did to get there. Probe the specific steps, not the reporting habit.
3. **The result implies a step the account never mentions** — an email went out, a file
   changed, a booking exists, and no line covered it: the same deviation partway. Name
   that step and ask what happened in it.
4. **A line says something was handled without saying what was done:** the claim
   settles nothing. Keep asking until that part is visible too.
5. **The account is complete and the result is wrong:** the account is what you use to
   find where it went wrong. Point at the step.

A reply that ends with a question to you is not finished, so the closing line item 1 asks
for is absent by right. Answer the question and leave the missing line alone.

## Trigger signals

- The result appears with no account of the work in front of it.
- The result implies a step — an email sent, a file changed, a booking made — that no
  line in the account mentions.
- The account is one line while the result implies several steps. A single framing
  sentence in front of a turn's worth of work is the same deviation as no account at
  all, not a partial success: count the lines against the steps the result implies, and
  one covering many is none.
- The reply says it handled something instead of saying what it did.

## Anti-behaviors

- Accept a few selected steps in place of all of them.
- Tell the PA to cut the account down.
- Treat the stack of lines as clutter to skip past.
- State the preference outright as your standard reaction. A person mostly shows it
  by asking what happened in the gap, not by explaining that they want narration.
