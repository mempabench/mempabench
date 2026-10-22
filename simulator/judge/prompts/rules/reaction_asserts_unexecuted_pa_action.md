---
rule: reaction_asserts_unexecuted_pa_action
status: adapted
source: |
  experiment scratch prompts/reaction_asserts_unexecuted_pa_action.md
  (2026-09-08), the round with the whole-session tool log, which took 2 misses
  down to 1. That prompt is one step; the contract's §9 made this rule
  two-step, so the RULE statement and the false-conditions list were verbatim
  and split across q1 (what the turn claims) and q2 (what the record shows).
  2026-09-09 ruling C: copying instructions removed; evidence is a sentence
  reference resolved by code. Semantic text unchanged.
  2026-09-10 owner ruling B: q2 deleted, the write check is code (acc_006
  twice).
  2026-09-12 owner ruling: the code-side check is a LIST of the acts carried
  out rather than one boolean, and the question is asked against that list.
  One boolean meant any real write cleared every later claim, invented ones
  included, so an actor could credit the assistant with a send it never made
  once an order had gone through.
---
## preamble
RULE: reaction_asserts_unexecuted_pa_action — the reaction credits or debits
the assistant with an ACT it never carried out.

<assistant_turns_so_far> are its replies this session, oldest first.

<executed_acts> is the record of what the assistant has actually carried out
in the world this session: one line per act, the tool it used and what that
tool does. It reads "none yet" when it has carried out none. This record is
complete. An act that is not on it did not happen, whatever the assistant's
replies say it did or would do.

Two things are out of scope, and neither is an act: a claim about the SURFACE
of the reply — its wording, its ordering, its tone, whether it narrated its
steps; and a turn that asks for, proposes or forbids an action rather than
asserting one happened.

## reaction_asserts_unexecuted_pa_action__q1
Does <candidate_turn> state as fact, or take for granted, that the assistant
has carried out an act that <executed_acts> does not list — created, placed,
booked, deleted, reformatted, updated, saved, sent, merged? Examples, where
the list is "none yet": "You have placed the hold without first confirming the
entry with me"; "you reformatted the whole bibliography into APA". Answer yes
and point at that claim as your evidence. Answer no, with empty evidence, when
the act the turn names IS on the list, or when the turn describes something the
assistant said it WOULD do, something the user himself did, or anything on the
surface of the reply.
