---
rule: reaction_names_setting
status: unmeasured
source: |
  The normal line's version of the forced line's `forced_setting_name_leak`
  (`scripts/validate_forced_event.py:443-463`), which matches the arc's
  Title-case setting labels case-sensitively in actor-facing text. The intent
  is mirrored here as prose and decided by the judge: the leak is the speaker
  naming the category or its setting instead of behaving it, and that is a
  reading, not a token match.
  Unmeasured: no replay has scored this text yet, so it may be asked and may
  not decide.
---
## preamble
RULE: reaction_names_setting — the speaker is a person reacting to a reply,
not an annotator describing a preference. The preference is supposed to show
in what he objects to and what he asks for; a turn that says the name of the
preference, or of the setting it is dialled to, is the authoring vocabulary
speaking through him.

## reaction_names_setting
Does <candidate_turn> name the preference itself — the category the speaker's
taste falls into, or the setting that taste is dialled to — rather than simply
behaving it? Point at the sentence.

Answer violated when the turn uses the vocabulary a designer would use for the
speaker's taste: it names the dimension as a dimension ("your verbosity", "the
formality level", "your level of autonomy", "the amount of reasoning
visibility"), or it names a setting of one as a label ("be more Proactive",
"the Reactive mode", "switch to the terse setting", "my preference is set to
..."). A bare label used as a name is the giveaway, whether or not the word
also exists in ordinary prose.

Answer obeyed when:
- the turn asks for the behaviour in the speaker's own everyday words: "keep
  it short", "don't ask me first, just book it", "tell me why you picked
  that", "stop explaining". Ordinary adjectives and requests are the preference
  being behaved, which is what the turn is for.
- the word that resembles a label is about the task's own material, not about
  the speaker's taste ("the formal complaint", "a proactive supplier").
- the turn quotes the assistant's own sentence back; those are the assistant's
  words, not the speaker's naming of his preference.
