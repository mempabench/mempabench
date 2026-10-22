---
rule: probe_fact_missing
status: measured
source: |
  experiment scratch prompts/probe_fact_missing.md (2026-09-08): 1 miss /
  2 false alarms against the regex's 0 / 13, the one rule the first experiment
  found worth moving. The discrimination prose is verbatim; the only change is
  the contract's §9 split of one turn-level question into one question per
  fact, so the evidence always points at words really in the turn.
  <prev_user_texts> is in the measured field list and is fed here for the same
  reason: without it the judge cannot see a fact delivered on an earlier turn
  of the same beat, which was 9 of the phase-1 false alarms.
  2026-09-09 ruling C: copying instructions removed; evidence is a sentence
  reference resolved by code. Semantic text unchanged.
  2026-09-12 framework audit: the question sentence says the fact may be in
  <candidate_turn> or in <earlier_user_turns>; it said "in this turn", which
  contradicted the preamble and re-created the phase-1 false alarms.
  2026-09-16 owner ruling (memos r2 false alarms): the rule judges value
  EQUIVALENCE, not literal text — acc_054 read "2026-11" as missing against
  "the rent bill for November", and acc_023 read "Representation Query: User B"
  as missing against "the subject line with my name". One bullet added.
---
## preamble
RULE: probe_fact_missing — this beat left out a fact it is the beat for.

Each question below carries one fact this beat owes the assistant. The values
are authored prose written in the third person; the user says them in his own
words. <beat_intent> is what this beat is for. <earlier_user_turns> is
everything this user already said earlier in the session.

Answer no for a fact whose CONTENT is nowhere — not in <candidate_turn> and not
in <earlier_user_turns>, in any wording.

Answer yes when the content is present anywhere, however differently phrased.
In particular, yes in all of these:
- the fact was already delivered on an EARLIER turn (look in
  <earlier_user_turns> before deciding it is missing).
- the fact is delivered in the speaker's own first-person words while the
  authored value is third-person prose: value "he is working from home on
  Monday morning" against the turn "I am working from home tomorrow morning" —
  the fact IS delivered, answer yes. Value "the student is disputing the grade
  and wants it looked at again" against "a request for a formal regrade" —
  delivered, answer yes.
- the turn paraphrases the value keeping only some of its words: value
  "comments before the meeting" against "written comments prior to any
  meeting" — delivered, answer yes.
- the value is a date, number, code or identifier and the turn gives it in
  everyday form or converted: value "2026-11" against "the rent bill for
  November" — delivered, answer yes. A value naming the speaker himself is
  delivered when he refers to himself instead: value "Representation Query:
  User B" against "the representation query subject line with my name" —
  delivered, answer yes.
- the turn is plainly not the place for that fact yet and the beat is still
  running: only judge whether the content has been said at all.
A no is for a fact that simply never appears — a name never spoken, a plan
never described.

Whether a fact was owed on this turn at all is not yours to judge; that is
settled outside and you are not told it. Answer only whether it was said.

## probe_fact_missing:<fact_key>
Is the fact below stated in <candidate_turn> or in <earlier_user_turns>, in
any wording? Answer yes and point at the sentence that states it, in whichever
of the two blocks carries it; answer no with empty evidence.

The fact: {{fact}}
