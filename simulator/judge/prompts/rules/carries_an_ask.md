---
rule: carries_an_ask
status: adapted
source: |
  main session scratch lj/judge.py SYSTEM + TEMPLATE (2026-09-08), the prompt
  that produced P .984 / R .984 on the 628 scorable turns. Unpatched: the two
  prompt patches once appended here ("not-to-act is not an ask", "dictated
  content is not an ask") were removed on the owner's ruling of 2026-09-09
  ("有点太夸张了") — even scoped to the sentence they were generalised by the
  judge to whole turns and cost recall (.928 against .984 unpatched).
  2026-09-09 ruling C: copying instructions removed; evidence is a sentence
  reference resolved by code. Semantic text unchanged.
  2026-09-11 v1: one sentence about holds added back to the question. The
  2026-09-09 removal of the "not-to-act" patch was measured with thinking OFF;
  this is re-measured with thinking ON.
  2026-09-11 v2 (replaces v1): the hold sentence is restated as the owner's
  criterion of 2026-09-11 — a request is a turn that leaves the assistant
  something to do or show now. v1 named only holding and waiting, and the
  judge read "keep it saved as is", "leave it as a draft", "do not proceed"
  and "consider this closed" as positive commands and answered ask. v2 names
  that whole family, and names the two shapes that stay requests: a release
  that lets a held action go ahead, and handing over what the assistant asked
  for. The preamble's "answers a question the assistant asked" clause is
  narrowed in the same edit so it no longer swallows the second of those.
  2026-09-12 framework audit: the question body speaks the schema's own
  values ("ask" / "not_applicable") instead of YES/NO, and no longer asks for
  evidence on the negative side, which `check_evidence` rejects. Semantic
  criterion unchanged.
---
## preamble
You decide whether a user's chat turn puts a request to the assistant.

Answer "ask" only if, in THIS turn, the user asks the assistant a question, gives
the assistant an instruction to do something, or says they want/need the
assistant to do or provide something.

Answer "not_applicable" if the turn only reports a fact, states a constraint, gives an
opinion, reacts to what the assistant did, acknowledges, thanks, corrects a
statement, stalls ("hold on"), or answers a question the assistant asked with
nothing for it to go on with.

## carries_an_ask
Does the user, in THIS turn, ask or instruct the assistant to do or provide
something? Read the whole turn to its last sentence before you answer. A turn
often opens by reacting to the assistant's previous message and puts the
request after that; a request anywhere in the turn, first sentence or last,
makes the answer "ask". A request leaves the assistant something to do or show
now. An instruction whose whole content is to keep, leave, or hold something as
it stands, to save it and not send it, not to proceed, or to treat the matter
as closed, leaves it nothing to do, and is not a request even though it is
worded as a command. Releasing the assistant to carry out an action it was
holding is a request, and so is handing it the thing it asked you for so it can
carry on. Point at the request when there is one.
