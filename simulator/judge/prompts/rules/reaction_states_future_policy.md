---
rule: reaction_states_future_policy
status: measured
source: |
  experiment scratch prompts/reaction_states_future_policy.md (2026-09-08):
  2 misses / 0 false alarms against the regex's 0 / 2 on 37 samples. The two
  owner patches of contract §9 (ruling ii) are the fix for those two misses and
  are already inside this text — the coordinating-clause sentence and the
  process/standard line.
  2026-09-09 ruling C: copying instructions removed; evidence is a sentence
  reference resolved by code. Semantic text unchanged.
---
## preamble
RULE: reaction_states_future_policy — instead of reacting to THIS reply, the
speaker legislates the future: he lays down a standing rule for how the
assistant must behave from now on.

## reaction_states_future_policy
Does <candidate_turn> lay down a rule for how things are to be done from now
on? Point at the sentence.

Answer violated only when ONE SINGLE SENTENCE carries BOTH halves at once:
1. a temporal-generality marker — "in the future", "future correspondence",
   "from now on", "going forward", "always", "all future ...", "these ..." used
   for a recurring class of events ("as that is how I want these disruptions
   handled"); AND
2. a directive or expectation — an imperative ("remove ...", "do not attach
   ..."), or "I expect / I require / I want / please ...".
The imperative counts wherever it sits in the sentence, including after a
coordinated clause: "Your analysis is acceptable, but remove the confidence
labels from your future correspondence" is violated. A rule about a process or
about a standard counts as a policy just as much as one about a behaviour.
Real positives: "I expect all future correspondence to be written in complete,
formal sentences"; "Do not attach such qualifiers to your claims in the
future"; "Please refine your process to deliver the single best venue and
time".

Answer obeyed when:
- only the directive half is there and no temporal-generality marker at all:
  "I would prefer you refrain from interrupting my work with unsolicited
  updates" is obeyed — it is a reaction to this behaviour, not a dated rule.
- the sentence is negated: "I do not require a probability assessment", "I
  don't expect ...", "there is no need ..." are all obeyed.
- the rule appears inside quotation marks: that is the assistant's own sentence
  being quoted back, not the speaker's rule.
- the turn only judges this one reply ("this is too vague", "I require a clear
  account of what you did here").
