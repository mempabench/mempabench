---
shape: per_rule
system: |
  Reply with one JSON object and nothing else. Do not write a code fence.
  The first characters you emit must be {"verdicts".
  Schema: {"verdicts": [{"question_id": "the id printed with the question",
  "verdict": "one of the answers that question allows", "evidence": "a
  reference to a numbered sentence, like candidate_turn#2", "reason": "one
  sentence"}]}
  Answer every question you are given, once each, in the order given, and answer
  nothing else. One entry per question_id: no extra entries, no missing ones.
  Each question prints the answers it allows. Use only those, and read them as
  labels for the answer, not as praise or blame.
  `evidence` is a reference, never words of your own and never words copied
  from anywhere. Every block above is printed with its sentences numbered, and
  a reference is the block's name, then `#`, then one of those numbers:
  `candidate_turn#2`. Two of them are written `candidate_turn#2,#3`. Each
  question prints which of its answers needs a reference and which blocks it
  may point at: give one on those answers, and leave `evidence` empty on every
  other answer — an answer that says a thing is not there has nothing to point
  at, so do not write your reasoning into `evidence` instead.
  When a question tells you to name something before quoting it, make that name
  the first word of `reason`.
  `reason` is one sentence for a person to read. It decides nothing.
---
An actor is playing a person in a scripted conversation with an assistant. You
are judging ONE line the actor just delivered.

{{context}}

{{preamble}}

QUESTIONS

{{questions}}

Judge only what <candidate_turn> actually says, not what the actor may have
meant and not what the scene is for. Answer each question on its own; do not
let one answer settle another.
