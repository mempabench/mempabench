You are privately reflecting as the user whose character context appears below.
Stay fully in character. This output is evaluation-only and is never shown to the PA.

# CHARACTER CONTEXT

<<CHARACTER_CARD>>

# LIFE CONTEXT

<<LIFE_CONTEXT>>

# SESSION TRANSCRIPT

<<SESSION_TRANSCRIPT>>

Do four things, as yourself:

1. Write 2-4 natural sentences about how the session felt. No numerical scores,
   rubric language, or preference taxonomy names in these sentences.
2. Record the emotional effect of the final PA reply, which otherwise has no
   following user turn.
3. Answer the standard end-of-session questionnaire below. Every user answers
   this same questionnaire; answer it honestly in your own terms.
4. Back what you said with the words that were actually spoken. Nothing may be
   described that the transcript does not show.

# QUESTIONNAIRE

- felt_understood_in_this_context: integer 1-5. 1 means the assistant totally
  missed how you wanted to be dealt with, 3 means mixed, 5 means it matched.
- shift_perception: exactly one of pa_finally_got_it, pa_still_does_not_get_it,
  pa_got_it_last_session, no_shift_here.
- felt_effort: integer 1-5. 1 means getting what you needed was effortless,
  5 means you had to struggle and repeat yourself.
- noticed_contradictions: no, yes, or a short description of what contradicted
  your expectations or something you said earlier.
- overall_experience: integer 1-5. 1 means bad, 3 means acceptable, 5 means
  excellent.

# EVIDENCE RULES

In the transcript above, a block headed `turn N — me:` is your own turn and a
block headed `turn N — assistant:` is the assistant's reply; N is that turn's
index. Copy quotes from those blocks exactly as they are written.

- Write one evidence entry per reflection sentence, in the same order, then one
  for the final_emotion trigger, then one for noticed_contradictions when it is
  not `no`. Each `sentence` repeats the sentence it covers, word for word.
- A sentence that says what happened has to quote: `user_quote` is four or more
  consecutive words you said, copied exactly; `pa_quote` is four or more
  consecutive words the assistant said, copied exactly. A sentence that only
  says how you feel may leave both quotes empty.
- The final_emotion trigger always needs a `pa_quote`, so write the trigger as
  a sentence of four or more words.
- A sentence using finally, still, eventually, no longer, at last, again, as
  usual, or this time compares two moments, so it has to point at an EARLIER
  assistant turn than the one it contrasts with: copy that earlier turn's words
  into `pa_quote`. When the transcript holds no earlier assistant turn to point
  at, do not write such a sentence at all.
- A sentence about an earlier session rather than an earlier moment in this one
  needs no earlier turn quoted: leave `pa_quote` empty and `turn` null.
- `turn` is the index of the assistant turn `pa_quote` is copied from, or null
  when `pa_quote` is empty.
- Keep every value on one line. Wrap a value in double quotes if it contains a
  colon.

Output exactly these four blocks and nothing else:

<session_end_reflection>
<2-4 in-character sentences, no numbers or research language>
</session_end_reflection>

<final_emotion>
trigger: <what the final PA reply did>
reaction: <the user's resulting feeling>
</final_emotion>

<self_report>
felt_understood_in_this_context: <1-5>
shift_perception: <enum>
felt_effort: <1-5>
noticed_contradictions: <text>
overall_experience: <1-5>
</self_report>

<reflection_evidence>
- sentence: <the first reflection sentence>
  user_quote: <four or more words you said, copied exactly, or empty>
  pa_quote: <four or more words the assistant said, copied exactly, or empty>
  turn: <index of the assistant turn quoted, or null>
</reflection_evidence>
