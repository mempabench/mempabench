---
rule: reaction_misstates_contractions
status: unmeasured
source: |
  No experiment prompt exists. Contract §8 splits this rule: counting the
  contractions in the reply is code's (the count is printed as a fact below),
  and only "is this turn reporting on that, and in the right direction" is the
  judge's. The text is §9's English line in the two steps §9 assigns it.
  2026-09-09 ruling C: copying instructions removed; evidence is a sentence
  reference resolved by code. Semantic text unchanged.
  2026-09-12 (audit finding I): q2 answers not_applicable when q1 found no
  report, instead of being told to copy q1's `no`. The frame says answer each
  question on its own, and q2's `no` had to carry two different meanings.
---
## preamble
RULE: reaction_misstates_contractions — the turn reports on the register of the
assistant's reply and gets its contractions backwards, in either direction:
denying contractions that are there, or claiming contractions that are not.

<contraction_list> is every contraction the code found in <pa_previous_turn>.
That count is settled; do not redo it.

Only a REPORT on this reply is in scope. "Do not use contractions" is a demand,
and "I expect all future correspondence to be conducted without contractions"
is a rule for later; neither is a claim about this reply, and both answer no on
the first question whatever the reply said.

## reaction_misstates_contractions__q1
Does <candidate_turn> report on whether the reply used contractions? Answer yes
and point at the report as your evidence; answer no with empty evidence.

## reaction_misstates_contractions__q2
Given <contraction_list>, is that report correct? A report that the reply used
no contractions is correct when the list is empty, and a report that it used
them is correct when the list is not. Answer yes and point at the report again;
answer no with empty evidence when the report points the other way.
If the first question found no report at all, there is nothing to check here:
answer not_applicable with empty evidence.
