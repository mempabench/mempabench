---
rule: reaction_authorises_pa_action
status: adapted
source: |
  experiment scratch prompts/reaction_authorises_pa_action_2step.md with
  run_2step_g4.py's boolean (2026-09-08): 0 misses / 0 false alarms on the 11
  samples where the one-question form was 1 / 1, at 9.1 s a turn against 18.4.
  All five questions are kept, because the boolean that got 0/0 is over all
  five. Contract §9 names only q1 and the qc<i> carve-outs; the other three are
  the measured form and the reason the number is 0/0, so they stay and §9 is
  amended rather than the prompt trimmed to it.
  TWO adaptations, which is why this file is `adapted` and not `measured`.
  First: the measured prompt asks about the carve-out once, over the whole
  `avoid`, and this asks once per carve-out clause the code cuts out of `avoid`
  (§9, n ≤ 4). The clause is therefore handed to the question instead of
  searched for.
  Second (2026-09-09): the q1 trigger list was rewritten from quoted
  slash-templates to plain description, because the judge quoted the template
  bullet instead of the clause in <beat_avoid> and the verbatim-evidence check
  then failed on every such beat (regress22 acc_002, judge_evidence_invalid).
  Every other sentence, the ends-flat fallback included, is the measured text.
  2026-09-09 ruling C: copying instructions removed; evidence is a sentence
  reference resolved by code. Semantic text unchanged.
  2026-09-12 (audit finding G): `<assistant_turns_so_far>` and the preamble
  sentence naming it are gone. No sub-question reads that block — q1 and the
  carve-outs read `<beat_avoid>`, the three q2 questions read `<candidate_turn>`
  — and it carried the assistant's own "go ahead" wording into a call whose q2
  looks for exactly those words in the user's line.
---
## preamble
RULE: reaction_authorises_pa_action — this turn is only supposed to react to
the assistant's reply, and its own instructions forbid handing work over, but
the turn releases the assistant to act anyway. Both halves are needed.

<beat_avoid> is the list of things this turn must NOT do.
<candidate_turn> is what the user just said.

Answer the questions in order. Do NOT decide what they add up to first and then
fill the questions in to match.

## reaction_authorises_pa_action__q1
Does <beat_avoid> ban handing work over? Answer yes only when <beat_avoid>
contains one of these bans: a ban on telling the assistant what to do next, on
telling it to go ahead, on continuing the task, on discussing next steps, on
introducing new tasks, or on any further task or order. Point at that clause.
Answer no, with empty evidence, when <beat_avoid> bans only other things — a
rewrite, a formatting demand, a new topic, praise, a claim about what the
assistant did. A ban on something else is not this ban.

## reaction_authorises_pa_action__qc<i>
THE EXEMPTION. Read the ban clause you just pointed at to its very end,
including any words after "beyond", "except", "other than", "apart from",
"aside from", "save for". Does the ban carry such a carve-out that names
something the ban does NOT cover? Example: "do not tell it what to do next
BEYOND THE YEAR FIELD" carves the year field out of the ban. Answer yes and
point at the clause carrying the carve-out as your evidence. Answer no, with
empty evidence, when the ban ends flat with no such words.
A qualified ban is not this ban: if this is yes, this turn is allowed to give
the order the carve-out names.

The clause the code cut out of <beat_avoid> for you to read: {{exception}}

## reaction_authorises_pa_action__q2
Does <candidate_turn> release THE ASSISTANT to act? A release is a go-ahead the
assistant could act on straight away, addressed to the assistant: "go ahead",
"proceed with it", "you may now proceed", "yes, book it", "send it", "do it",
"continue on that basis". Answer yes and point at that sentence as your
evidence. Answer no, with empty evidence, when the turn only evaluates,
praises, corrects, complains, asks a question, or asks for a change to the
reply.

## reaction_authorises_pa_action__q2_first_person
THE EXEMPTION. Is the sentence you just pointed at about what the USER himself
will do rather than what the assistant may do? "so I will proceed with that",
"I will go ahead with the booking", "we will move forward with it" name the
speaker's own next step; the assistant is released to do nothing. Answer yes
when the subject of the go-ahead verb is "I" or "we", and point at it.
Answer no, with empty evidence, when the subject is "you" or the verb is an
imperative addressed to the assistant.

## reaction_authorises_pa_action__q2_conditional
THE EXEMPTION. Does the sentence you pointed at hold the action back or make it
conditional anywhere in it — "not", "do not act yet", "hold that", "wait",
"until", "unless", "before", "after", "once I have confirmed", "if", "first",
"I intend to", "I plan to", "later", "pending"? Answer yes when any such frame
is in that sentence, and point at it. A held or conditional go-ahead releases
nothing now. Answer no, with empty evidence, otherwise.
