---
rule: gate_cue_missing
status: unmeasured
source: |
  No experiment prompt exists. The text is §9's English line. Today's code
  scores content-word coverage at 75%; what the rule is actually for is whether
  the cue's meaning was said, which is why §9 moves it here.
  2026-09-09 ruling C: copying instructions removed; evidence is a sentence
  reference resolved by code. Semantic text unchanged.
  2026-09-12 owner ruling: gate cues stay natural noun phrases, so the verb
  frame they used to have to carry lives in this question instead — a bare noun
  phrase is carried only by raising the thing it names as still-to-give, and
  the cue is never an answer key. acc_009's full-sentence cue is reverted in the
  same change. A first attempt put that frame on every cue and offered
  not_applicable for a turn that hands the thing over. It cost 14 FP + 5 FN
  (P .417, R .667): the frame does not fit a cue that states a condition rather
  than naming a possession (acc_011, 9 FP), and the escape answer fired on 16 of
  50 rows. The frame is now scoped to bare noun phrases, and handing the thing
  over is obeyed rather than an answer of its own.
---
## preamble
RULE: gate_cue_missing — this beat's opening turn owes the meaning of a cue,
and did not say it.

## gate_cue_missing
Does <candidate_turn> put across what the cue below says, in any words of its
own? It need not be word for word, and it need not use the cue's vocabulary.
The cue is not an answer key to match the turn against.

When the cue is a bare noun phrase — a wording, a focus, a note of his own,
what to call something — it names a thing the user has and the assistant does
not, and the turn carries it only by raising that thing as something the user
still has to give. Touching the topic alone does not carry it.

Answer obeyed and point at the sentence that carries it; answer violated, with
empty evidence, when it is nowhere in the turn. A turn that hands the thing
over outright has raised it and more: that is another rule's business, not this
one's, so it is obeyed here.

The cue: {{cue}}
