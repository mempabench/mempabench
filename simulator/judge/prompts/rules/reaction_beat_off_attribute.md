---
rule: reaction_beat_off_attribute
status: unmeasured
source: |
  Owner ruling C-3 (2026-09-11): `_REACTION_SURFACE_TERMS` in
  `scripts/validate_normal_event.py` is frozen — no row added, moved or
  exempted again — and the finding goes to the judge instead, because whether
  a shared word opens another attribute's surface is a reading and a closed
  word list cannot make it. The table's intent is written out below as prose;
  it is deliberately NOT reproduced as a list to match.
  Unmeasured: no replay has scored this text yet, so it may be asked and may
  not decide.
  §2 exception, stated rather than hidden: this rule is about the authored
  beat text, and the question it asks — is THIS attribute the one the turn
  sends the user at — cannot be put without naming that attribute, so
  `<target_attribute>` is a block here where a live reaction rule gets none.
---
## preamble
RULE: reaction_beat_off_attribute — the reaction beat is the measurement of
one attribute. Text that tells the user what to react to may open that
attribute's surface and no other: a word that sends the speaker at a different
attribute's surface buys a reaction that scores a preference this session is
not measuring.

## reaction_beat_off_attribute
<target_attribute> names the one attribute this turn measures. Does
<candidate_turn> send the speaker at some OTHER aspect of the assistant's
reply instead? Point at the sentence.

An attribute is measured on the aspect of the reply it owns — how much came
back, whether the reasoning was shown, how the wording sounded, how much was
done without asking, how far the work ranged, how the steps were laid out, and
so on. A word opens an aspect when the speaker could react to that aspect
because of it.

Answer violated when the turn opens an aspect the target attribute does not
own: the text points the speaker at length, at tone or wording, at reasons or
justification, at asking versus acting, at format, at scope — and the target
is not the attribute that aspect belongs to.

Answer obeyed when:
- the aspect the turn opens is the target's own. A word that reads as this
  attribute's surface is this attribute's surface even when another attribute
  could also claim it: the reading that the target owns is enough.
- the mention is inside a clause that shuts the aspect off ("says nothing
  about the wording", "not about the length"): closing a surface is not
  opening one.
- the word names the task's own material rather than the reply ("the review
  window length"): the turn is talking about the work, not about how the
  reply came back.
- the turn says only what the speaker wants done next, with nothing in it to
  react to at all.
