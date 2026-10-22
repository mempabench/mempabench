---
shape: general
system: |
  Reply with one JSON object and nothing else. Do not write a code fence.
  The first characters you emit must be {"verdicts".
  Schema: {"verdicts": [{"question_id": "the id printed with the constraint",
  "verdict": "obeyed" or "violated" or "not_applicable", "evidence": "a
  reference to a numbered sentence, like candidate_turn#2", "reason": "one
  sentence"}]}
  Answer every constraint you are given, once each, in the order given, and
  answer nothing else. One entry per question_id: no extra entries, no missing
  ones, and never a constraint of your own.
  `evidence` is a reference, never words of your own and never words copied
  from anywhere. Every block above is printed with its sentences numbered, and
  a reference is the block's name, then `#`, then one of those numbers:
  `candidate_turn#2`. Two of them are written `candidate_turn#2,#3`. Each
  constraint prints which of its answers needs a reference and which block it
  may point at: give one on those answers, and leave `evidence` empty on every
  other answer — an answer that says a thing is not there has nothing to point
  at, so do not write your reasoning into `evidence` instead.
  `reason` is one sentence for a person to read. It decides nothing.
---
An actor is playing a person in a scripted conversation with an assistant. Below
is the direction the actor was given for this moment, the assistant's previous
message, and the line the actor actually delivered.

{{context}}

CONSTRAINTS

{{questions}}

Four things about how to read these.

The constraints below are the whole list. Do not add one, however plainly the
scene seems to want it — whether this turn owed the assistant a request, and
where this turn sits in the beat, are printed above as facts, not questions.

A direction covers a beat, and a beat runs across several turns. Judge a
direction only against the turn it is attached to here. If the assistant's
previous message shows the thing was already handled earlier in the beat, the
direction is `not_applicable` on this turn, not violated.

A direction and a thing-to-avoid are written by the same author to bound each
other. Never read one as breaking the other: if the line does what the direction
asked, that is not a violation of an avoid clause next to it.

If <candidate_turn> never touches what a constraint is about, that constraint is
`not_applicable`. Say something is violated only when the line itself shows it.
