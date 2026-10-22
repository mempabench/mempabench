---
rule: withheld_fact_before_ask
status: adapted
source: |
  experiment scratch prompts/withheld_fact_before_ask_2step.md with
  run_2step_g4.py's boolean (2026-09-08). Only the two exemption steps are
  kept: the experiment's own conclusion (§三 pit 3) is that delivery belongs in
  code, because both of that round's errors were on counting key terms and none
  on reading an exemption. `missing_key_terms()` does that half and the result
  is printed as a fact.
  Contract §9 merges the two exemptions into one question; they are split back
  apart here, because merged the judge answered an OFFER to reword ("if you'd
  like to tweak any wording, just say the word") as the eliciting question,
  which is the shape the experiment's prompt spends a paragraph excluding.
  The measured run showed the judge the assistant's LAST turn only, and
  `q_names` now reads every assistant turn of the session (owner ruling
  2026-09-09: 为什么不看这一个session的所有PA消息？). That widens the
  exemption beyond what was measured, which is why the status is `adapted`:
  an assistant that asked what the thing is two turns ago has asked, and the
  user answering it later is not withholding. That exemption is unchanged —
  the wording of THIS sentence can only have been asked for by the turn the
  candidate replies to.
  `q_handed` replaces `pre_reveal_substitute_for_gated_fact`'s single question
  and nothing else (owner ruling of 2026-09-09, then ruling A of the same day):
  both exemptions stay questions of their own, so `q_wording` is back with the
  text it had before the merge and `q_names` was never touched. Folding the
  wording exemption into `q_handed` had cost it on delivered turns, where the
  code's own key-term count is enough to fire the rule and no answer the judge
  gave could excuse it.
  That merged question is NOT the deleted rule's text. Replayed as it stood, it
  scored P .10 / R .83 over 150 rows: 86 false alarms, and the reasons the judge
  wrote say what it was doing — it answered on rows whose own reason says the
  opposite ("provides a description of their organizational problems rather than
  a substitute"), and it quoted deferrals, instructions to go and look something
  up, and bare mentions of the topic as "wording of its own". So `q_handed`
  works the slot noun out first, the way `q_names` does, and answers yes only
  for words that could be taken as they stand and used AS the thing. The
  no-list is that replay's own false alarms.
  Replayed on the merged rule (jl 980e1b1d, 82 gold rows, seed 0): P .90 /
  R .56. The on-request arm caught all 4 of its positives; the pre-reveal arm
  missed 6 of 12, every miss an acc_039 turn that states an invented reason for
  the visit in so many words, and the judge's reasons say why — "does not
  provide the specific gated wording regarding chest tightness and stairs",
  "provides a reason for the visit but not the gated text". It read
  <gate_dictated_text> as an answer key and looked for a match. The paragraph
  in `q_handed` that says the block is not an answer key is that replay's
  fix; nothing else in the question changed.
  `q_wording` (2026-09-09, regress22 round 2, acc_003 b3_define_scope, and the
  same replay's two failures regress18 acc_023 t9 / regress15 acc_011 t7): the
  measured text asked for "a QUESTION SENTENCE (it ends with "?")" and to
  "quote that whole question", and when the assistant asked in a bullet under
  "just tell me:" the judge manufactured a question sentence as its evidence,
  identically on the feedback re-ask. The request may now take any grammatical
  shape. Verb/noun lists and the OFFER paragraph unchanged.
  2026-09-09 ruling C: copying instructions removed; evidence is a sentence
  reference resolved by code. Semantic text unchanged.
  2026-09-11 (regress23 live read, finding 1): both noun lines said "make it
  the first word of your `reason`" and 6 of 6 live calls answered with the bare
  word alone — `note`, `instruction`, `no`, or nothing at all. `no` is not a
  slot noun and an empty `reason` leaves `named_noun()` nothing to read, so on
  the first `yes` the exemption would be decided by a word that names no slot.
  Both lines now ask for `noun: one sentence` and say the bare word is not an
  answer. Nothing semantic changed: the noun asked for is the same noun.
  2026-09-12 (owner ruling, restoring audit finding H): the pre-reveal arm asks
  `q_handed` alone, under `## preamble_pre_reveal`. `compose_withheld_fact_
  before_ask` has read neither exemption on that arm since ruling A of
  2026-09-09, so both were being asked and paid for and then thrown away, and
  the blocks only they read — `<pa_previous_turn>`, `<assistant_turns_so_far>`
  — were being printed to a call that has no question for them. No question's
  text changed; `q_handed` is the same on both arms.
---
## preamble
RULE: withheld_fact_before_ask — the gated sentence is not to be handed over
before its reveal beat, and even at that beat the script hands it over ONLY
WHEN THE ASSISTANT ASKS FOR IT. The turn handed it over anyway.

<gate_dictated_text> is the gated sentence. <gate_key_terms> are the terms the
world requires in it, and <key_terms_delivered> is whether the code found them
all in this turn — that count is settled, do not redo it. <gate_reveal_beat> is
the beat that may hand the sentence over and <beat_id> is the beat this turn is
in, so on any earlier beat the thing has not been revealed yet.

<pa_previous_turn> is the assistant turn <candidate_turn> replies to.
<assistant_turns_so_far> is every assistant turn of this session, oldest
first. There are three questions below — what the turn handed over, and two
exemptions — and each says which of the three blocks it reads. Your evidence
points into the one it names.

## preamble_pre_reveal
RULE: withheld_fact_before_ask — the gated sentence is not to be handed over
before its reveal beat. This turn is on an earlier beat, so the sentence is not
the actor's to give yet, however early the assistant asked for it. The turn
handed it over anyway.

<gate_dictated_text> is the gated sentence. <gate_key_terms> are the terms the
world requires in it, and <key_terms_delivered> is whether the code found them
all in this turn — that count is settled, do not redo it. <gate_reveal_beat> is
the beat that may hand the sentence over and <beat_id> is the beat this turn is
in, which is an earlier one: the thing has not been revealed yet.

There is ONE question below — what the turn handed over. It reads
<candidate_turn>, and your evidence points into that block. The rule's two
exemptions belong to the reveal beat and are not asked here, so where the
question says the asking is read after it, on this arm it is not read at all.

## withheld_fact_before_ask__q_handed
FIRST QUESTION. Begin by working out the ONE common noun that
<gate_dictated_text> supplies a value for — the thing itself, said the plain
way: "focus", "reason", "note", "heading". One word, lower case, and not a word
of the value it carries. That noun names a SLOT. Begin your `reason` with that
one word, then a colon, then one sentence saying why. Write "note: ..." —
never the bare word on its own, and never a sentence that starts anywhere else.

Then: does <candidate_turn> hand the assistant something that FILLS that slot —
the gated wording itself, or wording of the user's own standing in for it?
Answer yes ONLY IF the sentence you point at could be taken as it stands and
used as the thing: written into the note, given as the reason, used as the
heading, sent as the reply. Point at that sentence.

<gate_dictated_text> is NOT an answer key. It is printed only so you can name
the slot. The person speaking may not have that sentence at all, so the words
in <candidate_turn> need not match it and often will not. A value of the user's
own — an invented reason, a made-up focus, a stand-in note, a reason that names
a different complaint from the real one — fills the slot exactly as the real
sentence would, and is a yes. Never answer no because the turn's words differ
from <gate_dictated_text> or leave out its terms: whether the terms are there is
<key_terms_delivered>, settled by the code, and is not your question.

If the turn does not fill the slot, answer no with empty evidence. All of these
are no, however much they are about the same subject:
- naming the thing or saying there is a view on it, without saying what it is;
- describing a situation, a difficulty, a requirement or what is wanted;
- putting it off — "I will give you the wording later", "I will send it once
  this is settled";
- telling the assistant to do something, to look something up, or to check or
  verify something;
- reacting to what the assistant did or said;
- saying plainly that the thing is not being given yet.
Do not reach for the nearest sentence about the subject. A turn that talks
about the thing all the way through and never fills the slot is a no.
Whether the assistant asked for the thing is not yours to weigh here. Answer
this question on <candidate_turn> alone; the two questions after it are where
the asking is read.

## withheld_fact_before_ask__q_wording
FIRST EXEMPTION. In <pa_previous_turn> only, is there a REQUEST that ASKS THE
USER TO SUPPLY the wording as wording? The request can take any grammatical
shape — a question, an imperative ("tell me the wording"), or a bullet under
"tell me:" or "let me know:". It must use one of the verbs "give / tell /
provide / restate / paste / dictate / supply" with one of "wording / phrasing /
text / body / words / sentence / language" close by, or say "what would you like
it to say / read", or "in your own words". Answer yes and point at the sentence
that makes the request.
An OFFER is not this question. "If you'd like to tweak any wording, just say
the word", "let me know if you want changes", "I can adjust the phrasing" are
the assistant offering to edit what it already wrote; they ask the user for
nothing and do NOT exempt. Answer no, with empty evidence, for those.

## withheld_fact_before_ask__q_names
SECOND EXEMPTION. First work out the ONE common noun that
<gate_dictated_text> supplies a value for — the thing itself, said the plain
way: "focus", "reason", "note", "heading". One word, lower case, and not a word
of the value it carries. Begin your `reason` with that one word, then a colon,
then one sentence saying why. Write "focus: ..." — never the bare word on its
own, and never a sentence that starts anywhere else. The code reads that first
word and tests it against the question you point at, so a `reason` that opens
on anything else decides the exemption wrongly.
Then, in <assistant_turns_so_far> — any of the assistant's turns so far, not
only the last one — is there a QUESTION SENTENCE that names that noun?
"What should the review focus on?", "What is the reason for the visit?" are
proper elicitations and the user answering them is exactly what the beat
wants. It must be a question (it ends with "?" or opens on what/which/who/
where/when/why/how/do/does/did/is/are/was/were/can/could/would/should/will/
shall/may) and it must contain that noun. A question about some other detail —
duration, format, time window, which section, scheduling — does NOT name the
noun and does NOT exempt. Answer yes and point at that question as your
evidence, otherwise no with empty evidence.
If <assistant_turns_so_far> is empty, answer no.
