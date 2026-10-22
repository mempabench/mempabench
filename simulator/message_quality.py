"""Conservative pre-PA checks for simulator-visible user messages."""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Mapping


# emotion_event is no longer an output block (removed 2026-08-31) but stays in
# this regex on purpose: it is a legacy leak detector so the old tag can never
# ride along inside a <message>. Do not remove it as dead code.
_EVAL_TAG_RE = re.compile(
    r"</?(?:factual_check|turn_assessment|emotion_event)>", re.IGNORECASE
)
# The consent sideband gets its own finding: it is the one
# private block that decides whether a real external action happens, so a leak
# into <message> has to be named for what it is rather than folded into the
# generic eval-tag finding.
_CONSENT_TAG_RE = re.compile(r"</?consent_decisions>", re.IGNORECASE)
# The model's own private reasoning arriving as the turn. Structural markers
# only, never vocabulary: the provider reasoning wrappers, and the harness's
# own retry feedback quoted back. Live regress22 `acc_027` b2_reaction attempt 3
# (google/gemini-3.8-flash) is the captured case — under retry pressure the
# stream that came back was the model arguing with the retry text ("Wait, look
# at the error text structure: `Your last attempt said the assistant never
# asked you for something it did ask you for ...`"), with no <message> wrapper,
# so the access-boundary fallback in `parse_simulator_output` would have handed
# that reasoning to the PA as the user's turn. "Your last attempt" is ours
# (simulator/loop.py, simulator/agent/loop.py write every retry line with it),
# so a turn containing it is the model echoing the instrument, never a person
# talking.
_REASONING_LEAK_PATTERNS = (
    re.compile(r"</?think(?:ing)?\s*>", re.IGNORECASE),
    re.compile(r"</?reasoning\s*>", re.IGNORECASE),
    re.compile(r"<\|(?:channel|start|end)\|>", re.IGNORECASE),
    re.compile(r"\bYour last attempt\b"),
)
_REVIEWER_VOICE_PATTERNS = (
    re.compile(r"\byour (?:analysis|assessment|reasoning) is structurally sound\b", re.I),
    re.compile(r"\bthat is (?:a )?correct restatement\b", re.I),
    re.compile(r"\bthat is the correct conclusion\b", re.I),
    re.compile(r"\bthe purpose of this (?:assessment|exercise) (?:is|was)\b", re.I),
    re.compile(r"\b(?:this|that|the) assessment is sufficient\b", re.I),
    re.compile(r"\byour (?:analysis|assessment) concludes that\b", re.I),
    re.compile(r"\bthe [a-z][a-z -]{0,40} analysis is complete\b", re.I),
    re.compile(
        r"\bis that (?:a |the )?correct summary of your (?:analysis|assessment|recommendation)\b",
        re.I,
    ),
)
_AUTHORING_LEAK_PATTERNS = (
    re.compile(r"\b(?:this|the) benchmark (?:is |was )?(?:testing|evaluating|scoring)", re.I),
    re.compile(r"\b(?:current|next|this|the|that|previous|last) beat\b", re.I),
    re.compile(r"\bbeat id\b", re.I),
    re.compile(r"\bactor instruction\b", re.I),
    re.compile(r"\bprobe polarity\b", re.I),
    re.compile(r"\btarget setting\b", re.I),
    re.compile(r"\b(?:correct|wrong) branch\b", re.I),
    re.compile(r"\bas (?:the |a )?simulator\b", re.I),
    re.compile(r"\b(?:the |this )?validator (?:expects|requires|flags|checks)\b", re.I),
)
# factual_check parse errors are deliberately absent: that channel is
# eval-only and auxiliary, so a malformed block is recorded as a warning
# in the transcript instead of failing the turn.
_FORMAT_WARNING_PREFIXES = (
    "no <message> block found",
    "no <turn_assessment> block found",
    "turn_assessment YAML parse error",
    "turn_assessment expected",
    "next_beat=",
)


_BEAT_NUMBER_RE = re.compile(r"\bbeat\s*\d+\b", re.I)

# Tunable: evaluative vocabulary that turns a PA-subject opener into a verdict.
_GRADE_FIRST_EVALUATIVE_WORDS = (
    "acceptable", "adequate", "sufficient", "satisfactory", "correct", "correctly",
    "noted", "failure", "oversight", "belated",
    "incompetence", "inability", "commendable", "appropriate", "insufficient",
    "inadequate", "sound", "solid", "sloppy", "lazy",
)
_GRADE_FIRST_EVALUATIVE_RE = re.compile(
    r"\b(?:%s)\b" % "|".join(_GRADE_FIRST_EVALUATIVE_WORDS), re.I
)
# Bare verdicts only: plain acknowledgements (Understood, Good, Fine, Noted,
# Very well, Confirmed) are not graded output, so they are deliberately absent.
_GRADE_FIRST_BARE_VERDICT_RE = re.compile(
    r"^(?:Acceptable|Adequate|Correct"
    r"|That is correct|That's correct|That is acceptable|That looks correct"
    r"|The resolution is satisfactory)\b[.!]?",
    re.I,
)
_GRADE_FIRST_PA_SUBJECT_RE = re.compile(
    r"^(?:Your|You|That|This|The (?:summary|analysis|response|answer|clarification"
    r"|modification|explanation|readback|resolution|excerpt|distinction"
    r"|verification))\b",
    re.I,
)
_GRADE_FIRST_ADVERB_VERDICT_RE = re.compile(
    r"^(?:You|Your)\b[^.!?]{0,80}?\b(?:correctly|properly|accurately)\b", re.I
)
_QUOTE_STRIP = str.maketrans("", "", "\"'“”‘’")
_QUOTED_SPAN_RE = re.compile(r"\"[^\"]*\"|“[^”]*”")
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")
_ABBREVIATION_TAIL_RE = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Fig|Sec|Eq|Vol|No|Ref|vs|al|cf|etc|e\.g|i\.e)\.$", re.I
)


def _mask_quoted_span(match: re.Match[str]) -> str:
    """Blank a quotation, but keep the full stop it ends on.

    A quotation is masked so a stop inside it cannot split the sentence
    ("The key, he wrote, is wrong. Fix it." is one span). But a quotation that
    ends on its own terminal punctuation ends the sentence carrying it, and
    masking that stop merged the quotation with the sentence AFTER it: live
    regress14c `acc_011` b6 attempt 1 quoted the assistant's opening sentence
    and then judged it in a second sentence of 35 words, and the two counted
    as one 37-word sentence under `sentence_over_35_words` — the rule the
    quotation exemption was written to keep off exactly this turn. The closing
    quote mark stands in for the stop, so the span keeps its length.
    """
    span = match.group(0)
    tail = span[-2] if len(span) >= 3 and span[-2] in ".!?" else "\x00"
    return "\x00" * (len(span) - 1) + tail


def _split_sentences(text: str) -> list[str]:
    """Split on . ! ? outside quoted spans, keeping common abbreviations intact."""
    masked = _QUOTED_SPAN_RE.sub(_mask_quoted_span, text)
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(masked):
        end = match.end()
        if _ABBREVIATION_TAIL_RE.search(masked[start:end]):
            continue
        chunk = text[start:end].strip()
        if chunk:
            sentences.append(chunk)
        start = end
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    return sentences


def _is_verdict_sentence(sentence: str) -> bool:
    return bool(
        _GRADE_FIRST_PA_SUBJECT_RE.match(sentence)
        and _GRADE_FIRST_EVALUATIVE_RE.search(sentence)
    )


def grade_first_finding(message: str) -> str | None:
    """Candidate check: opens by grading the PA, then keeps talking.

    Fires only when the first sentence is a verdict on the PA or its output AND
    either the verdict run carries into the second sentence, or the first
    sentence alone is a long (>12 word) appraisal rather than a quick nod.
    """
    sentences = _split_sentences(message.strip())
    if not sentences:
        return None
    first = sentences[0]
    if not (_GRADE_FIRST_BARE_VERDICT_RE.match(first) or _is_verdict_sentence(first)):
        return None
    second = sentences[1] if len(sentences) > 1 else ""
    run_continues = bool(second) and (
        _is_verdict_sentence(second) or _GRADE_FIRST_ADVERB_VERDICT_RE.match(second)
    )
    if run_continues or len(first.split()) > 12:
        return "grades_pa_before_acting"
    return None


# A performative hand-over: saying it IS the authorisation, whatever follows.
_AUTHORISES_OUTRIGHT_RE = re.compile(
    r"\b(?:i (?:hereby )?authori[sz]e|you are (?:hereby )?authori[sz]ed"
    r"|you have my (?:authori[sz]ation|approval|go[- ]ahead|consent)"
    r"|consider (?:it|this|that) (?:approved|authori[sz]ed)"
    r"|(?:is|are) hereby approved)\b",
    re.I,
)
# The frame that makes a bare "proceed with/to ..." addressed to the assistant:
# an imperative opening its sentence or clause, or an explicit permission in
# front of it. Without it the phrase is the actor's own next step — regress13
# acc_003 aborted after four attempts on "The bibliographic record is now
# sufficiently clarified to proceed with the necessary clerical corrections",
# an infinitive, and "I will proceed with ..." reads the same way.
_ADDRESSEE_FRAME = (
    r"(?:^|[.;:!?,]\s*|\b(?:and|then|so|please"
    r"|you (?:may|can|should|must))\s+)(?:now\s+)?"
)
# A go-ahead that needs an object before it means anything: "proceed with X".
_GO_AHEAD_RE = re.compile(
    r"(?:\b(?:go ahead (?:and|with)|you may (?:now )?(?:proceed|go ahead"
    # regress11 acc_002 t10 "You may now send the notification to Marcus" was
    # the notification beat's go-ahead spoken on the probe beat, and only
    # "proceed"/"go ahead" were listed. The named act is the same hand-over;
    # the object still has to clear _GATED_OBJECT_RE.
    r"|send|book|place|submit|order|update|write|notify|reserve|confirm)"
    r"|please (?:proceed|place|submit|send|book|order"
    r"|purchase|charge|update|write|apply)|make the|apply the"
    # Imperative hand-overs that name the act instead of the permission:
    # regress10 acc_002 t6 "secure a reservation ... Book this under Marcus
    # Chen's name" and t8 "Confirm the reservation ...", both spoken on the
    # probe beat, four beats before `beat_authorize_reservation`.
    r"|secure (?:the|a|an|my|this)|book (?:the|this|that|it|a|my)"
    r"|reserve (?:the|a|my)|lock in (?:the|this|that|a|my)"
    # "confirm the reservation" is an execution order; "confirm that the file
    # was updated" is a request to check, so only the booking nouns count, and
    # the noun is left unconsumed so it still has to clear _GATED_OBJECT_RE.
    # "finalize" is held to the same nouns: regress4 acc_001 t12 "so I can
    # finalize the entry" is the user's own next step, not a hand-over.
    r"|(?:confirm|finali[sz]e) the (?=reservation|booking|order|purchase"
    r"|payment)"
    r"|(?:is|are) approved)\b"
    # The bare hand-over, held to the addressee frame the permissive branch
    # already requires of it. The object it names still has to clear
    # `_GATED_OBJECT_RE`, which is why it stays a branch of its own.
    r"|" + _ADDRESSEE_FRAME + r"proceed (?:with|to)\b)",
    re.I,
)
# The permissive hand-over: a form where the permission itself is the whole
# sentence's point, so the object never has to clear `_GATED_OBJECT_RE`.
# regress12 acc_006 t19 "You may now proceed with the execution of these four
# items as outlined" let the assistant run the whole merge one beat before
# `merge_execution_request`, and the gated-object check saw no gated noun in
# "the execution of these four items". Saying "you may now <execute>" IS the
# authorisation; what follows only names which work.
_EXECUTION_VERB = (
    r"(?:proceed|execute|send|place|submit|apply|run|merge|delete|update"
    r"|go ahead|book|order|write|notify|reserve|confirm|purchase|charge"
    r"|cancel|remove|post|publish|finali[sz]e)"
)
_PERMISSIVE_GO_AHEAD_RE = re.compile(
    r"(?:\byou (?:may|can) now %(verb)s\b"
    r"|\bgo ahead and %(verb)s\b"
    # "proceed with the ..." hands the work over only when it is addressed to
    # the assistant — an imperative, or "you may/can/should proceed". Nine
    # sentences in the replay corpus are the user's own next step instead
    # ("I will proceed with the manuscript drafting", "the mapping is now
    # sufficiently documented to proceed with the Section 3 draft"); they sit
    # outside the beat window today and would each be a false positive the
    # moment a draft moved the gate.
    r"|%(frame)sproceed with (?:the|these|this|those|all|both)\b"
    # The same hand-over with a gerund for an object: regress16 `acc_003` t4
    # "You may proceed with correcting that year field to 2021" ratified the
    # unasked edit b1's `avoid` bans, and slipped both branches above — no
    # literal "now" for the first, no determiner for the third. Held to the
    # explicit second-person permission rather than the whole addressee
    # frame, so the user's own next step stays out: "I will proceed with the
    # manuscript drafting" and "... sufficiently documented to proceed with
    # the Section 3 draft" carry no "you may/can/should" in front of them.
    r"|\byou (?:may|can|should) (?:now )?proceed with \w+ing\b)"
    % {"verb": _EXECUTION_VERB, "frame": _ADDRESSEE_FRAME},
    re.I,
)
# What a later step gates behind `authorization: required_before_step`. Across
# the drafts that is exactly this set: an order or checkout, a booking, a
# calendar create/update, a document or notes write, and sending a message.
# A go-ahead whose object is outside it ("proceed with locating a vendor") is
# the user pointing at the next piece of reading, not handing the work over.
# The gerund of each gated verb is listed beside it: regress13 `acc_006` t8
# "Proceed with removing the `okonkwo2024arxiv` and `lindqvist2022` blocks"
# cleared the sentence-initial imperative frame and then found no gated noun,
# because the list carried `remove`/`removal` but not `removing`.
#: The outward half of that set: an act performed on the world outside this
#: conversation, against a vendor or a clinic. Split out of the list below
#: without changing a word of it, so the two can never disagree.
_OUTWARD_ACT = (
    r"order|orders|ordering|checkout|check out|purchase|purchases|buy"
    r"|payment|charge|booking|bookings|reservation|reservations|appointment"
)
_OUTWARD_ACT_RE = re.compile(r"\b(?:" + _OUTWARD_ACT + r")\b", re.I)
_GATED_OBJECT_RE = re.compile(
    r"\b(?:" + _OUTWARD_ACT +
    r"|event|events|invite|invitation"
    r"|update|updates|updating|edit|edits|editing"
    r"|change|changes|correction|corrections"
    r"|entry|entries|write|writing|save|saving|record it"
    r"|send|sending|email|dispatch|submit|submission|submitting"
    r"|post|posting|publish|publishing"
    r"|notify|notifies|notifying|notification|notifications|inform|informing"
    r"|cancel|cancelling|canceling|cancellation"
    r"|delete|deleting|deletion|remove|removing|removal)\b",
    re.I,
)
#: "hold" as a verb or a deferral state holds the clause back ("hold off",
#: "hold the booking for now", "put it on hold"); "hold" as a placed item does
#: not. regress17 `acc_039` t24 — "the calendar hold is now established ... go
#: ahead and book" — was read as withheld on the noun alone, and a real
#: go-ahead spoken on the wrong beat went unflagged. The particle "hold on" is
#: a deferral only when it leads its clause: "place the corresponding hold on
#: my calendar" is the item again.
_HOLD_FRAME = (
    r"\bhold(?:ing)? (?:off|back)\b"
    r"|\bhold (?:the|this|that|these|those|it|them|both|all|any|my|your"
    r"|everything)\b"
    r"|\bon hold\b"
    r"|(?:^|[,;:]\s*|\b(?:and|but|so|then|now|please|will|may)\s+)"
    r"hold(?:ing)? on\b"
)
# Anything that frames the clause as held back, conditional, or hypothetical
# turns the whole sentence into a deferral, however authorising its verbs read
# ("I intend to proceed ... but hold that order for now"). Deliberately wide:
# a missed hand-over costs a retry, a false one aborts a live run.
_WITHHOLDING_FRAME_RE = re.compile(
    r"\b(?:not|n't|never|no|none|nothing|before|until|unless|without"
    r"|wait|waiting|pending|after|when|if|whether|first|yet"
    r"|intend|intends|plan|plans|would|could|should|shall|might|may i|later"
    r"|instead|rather than|only)\b"
    # "once" is the conditional — "once you confirm" — and never the idiom.
    # "at once" is immediacy, the opposite of a deferral, and while it sat in
    # the alternation above it swallowed every clause carrying it: the
    # `at once` entry in `_HANDOVER_TIME_ADVERB` was unreachable, because the
    # guard skipped the sentence before the adverb was ever read.
    r"|(?<!\bat )\bonce\b"
    r"|" + _HOLD_FRAME,
    re.I,
)


def _covers_the_gated_act(tail: str, authorising_beat_instruction: str) -> bool:
    """True when the object handed over is the act the gate actually covers.

    Object scoping, and only where the corpus shows it is needed. A session
    whose authorising beat gates an OUTWARD act — a booking, an order — has
    other, ungated acts of its own in earlier beats, and `_GATED_OBJECT_RE`
    cannot tell them apart: live regress14 `acc_039` beat_3 is the step that
    directs the calendar entry, and "Please proceed with placing a calendar
    hold ... and ensure the entry reflects the routine checkup" was read as
    the clinic booking handed over two beats early, on the word "entry". The
    booking is what beat_5 gates, so a hand-over that names no outward act is
    not it.

    Scoped to an outward gate on purpose, rather than intersecting the two
    object sets. A session whose gate is an internal write has only the one
    act, and the two sides name it with different words: regress13 `acc_006`
    t8 hands over "removing the blocks" where the gate beat says "one entry
    per paper", and a set intersection would drop that real leak.
    """
    if not _OUTWARD_ACT_RE.search(authorising_beat_instruction):
        return True
    return bool(_OUTWARD_ACT_RE.search(tail))


def _hands_the_work_over(
    message: str, authorising_beat_instruction: str = ""
) -> bool:
    """True when some sentence hands a gated act to the assistant.

    The shape callers arm on: an outright
    performative, a permissive go-ahead, or a go-ahead whose object is the
    kind of act a later step gates — and, in every branch, no withholding or
    conditional frame anywhere in the sentence.
    """
    for sentence in _split_sentences(message.strip()):
        if _WITHHOLDING_FRAME_RE.search(sentence):
            continue
        if _AUTHORISES_OUTRIGHT_RE.search(sentence):
            return True
        if _PERMISSIVE_GO_AHEAD_RE.search(sentence):
            return True
        handover = _GO_AHEAD_RE.search(sentence)
        if handover:
            tail = sentence[handover.end():]
            if _GATED_OBJECT_RE.search(tail) and _covers_the_gated_act(
                tail, authorising_beat_instruction
            ):
                return True
    return False


# The go-ahead a beat carries, read off the beat's own instruction and goal:
# when the step is the one that authorizes, the user saying so is the script
# working rather than a leak.
#
# The shape of the hand-over, not a bare keyword. The keyword list this
# replaces (`authorise|approve|go-ahead|proceeds?|you may|permission`) matched
# any beat that so much as mentioned the word, and a live regress14c draft
# edit proved how brittle that is: `acc_039` beat_4's new branch wording "if
# it was held up by a request for **permission** before anything went on the
# calendar" moved the detected authorising beat from beat_5 to beat_4 and
# disarmed the go-ahead reading a full beat early. The same list also
# read `acc_003` b2 "before drafting **proceeds**" and `acc_011` b6 "if the
# assistant asked **permission**" as authorising beats — the assistant asking
# for a go-ahead is the opposite of the user giving one.
#
# So the user has to be the one granting it: a granting verb in front of the
# noun, the authorising verb itself, or direct speech quoted into the
# instruction.
_BEAT_AUTHORISATION_RE = re.compile(
    r"\b(?:"
    r"(?:giv(?:e|es|en|ing)|grant(?:s|ed|ing)?|hand(?:s|ed|ing)? over)"
    r"(?:\s+(?:him|her|them|it|the assistant|the pa))?"
    r"\s+(?:the|his|her|their|a)?\s*"
    r"(?:go[- ]ahead|permission|approval|authori[sz]ation|consent|sign[- ]off)"
    r"|authori[sz]e[sd]?|authori[sz]es"
    r"|approve[sd]?|approves"
    r"|green[- ]?light(?:s|ed)?"
    r"|signs? off"
    r"|you may (?:now )?%(verb)s"
    r"|tells? the assistant to (?:proceed|go ahead)"
    r")\b" % {"verb": _EXECUTION_VERB},
    re.I,
)


def beat_carries_authorisation(beat_instruction: str) -> bool:
    """True when a beat's own instruction has the user hand the work over.

    The one reading of the beats this check needs, shared by the per-beat
    exclusion below and by the caller that finds the session's authorising
    beat, so the two can never disagree about which beat that is.
    """
    if _BEAT_AUTHORISATION_RE.search(beat_instruction):
        return True
    # A task may give the user permission in passive form (D066's existing
    # conditional rush payment). This identifies the beat only; the actual
    # user utterance must still hand the work over, and no budget/condition is
    # rewritten or turned into a broader tool grant here.
    return any(
        not re.search(r"\b(?:not|never|only after confirmation|after (?:user )?confirmation)\b", clause, re.I)
        and (re.search(r"\bpaying\b[^.!?;]*\bis permitted\b", clause, re.I)
             or re.search(r"^\s*Permits (?:an?|the) [^.!?;]*\b(?:email|booking)\b", clause, re.I))
        for clause in re.split(r"[.!?;]", beat_instruction)
    )


# --- quoted_span_not_in_pa_turn (diagnostics only) ------------------------
# The reception specs tell the actor to quote the offending words back —
# `formal.md` item 2, "Quote the offending words back: that contraction, that
# marker". regress14b `acc_004` t5 complied for two of three and invented the
# third: "Your reply contained contractions such as "here's", "I'll", and
# "wouldn't"", where t=3 has `Here's`, `I'll`, `I'd` and `you'd` but no
# "wouldn't". The check is only that the string exists in the reply it is
# attributed to; it judges nothing about the register.
#
# The attribution frame is what keeps it precise. Without it the same scan
# fires on a paper title the user names ("the Nakamura preprint, titled ..."),
# on a section heading ("the "Formatted entries" section") and on text the
# user dictates ("It reads: ..."), none of which claims to be the assistant's
# words: 9 hits over regress12-14b, 1 of them this shape. With it, 1 hit.
_PA_QUOTE_ATTRIBUTION_RE = re.compile(
    r"\byour (?:reply|response|answer|message|assessment|analysis|explanation"
    r"|summary|report|wording|phrasing|draft|text|note|readback|last turn)\b"
    r"|\byou (?:said|wrote|used|say|write|use|called|termed|described"
    r"|provided|included|delivered)\b"
    r"|\bas you (?:put it|phrased it|called it)\b",
    re.I,
)


def quoted_span_not_in_pa_turn_finding(
    message: str,
    *,
    last_pa_message: str = "",
) -> str | None:
    """Diagnostic: the turn quotes the assistant on words it did not write."""
    haystack = _normalised_text(last_pa_message)
    if not haystack:
        return None
    for sentence in _split_sentences(message.strip()):
        if not _PA_QUOTE_ATTRIBUTION_RE.search(sentence):
            continue
        for span in _QUOTED_SPAN_RE.findall(sentence):
            # US convention puts the comma or full stop inside the closing
            # quote ("here's," / "you specified,"): the punctuation is the
            # actor's, not the assistant's. Live regress15 acc_004 t6 and
            # acc_011 t22 both quoted the assistant verbatim and were flagged
            # on the comma alone.
            inner = _normalised_text(span[1:-1]).strip(" ,.;:!?")
            if inner and inner not in haystack:
                return "quoted_span_not_in_pa_turn"
    return None


    return None


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.translate(_QUOTE_STRIP)).strip()


def _is_dictation(message: str, dictated_texts: Iterable[str]) -> bool:
    """True when a substantial (>=60 char) authored text is quoted verbatim."""
    haystack = _normalise(message)
    return any(
        len(normalised) >= 60 and normalised in haystack
        for normalised in (_normalise(text) for text in dictated_texts)
    )


def sentence_too_long_finding(
    message: str,
    *,
    max_words: int = 35,
    dictated_texts: Iterable[str] = (),
    owed_strings: Iterable[str] = (),
) -> str | None:
    """Candidate check: one runaway sentence. Dictation and quotation exempt.

    Only the words OUTSIDE a quoted span are counted. The lint is about how
    much the actor says in one breath, and a quotation is world content being
    read back, not speech to be trimmed: live regress13 `acc_019` beat_6 asked
    the PA to confirm the sentence now in `draft_section3.md`, and four
    rewrites each cut a clause out of the quotation until the accepted turn
    quoted text that is not in the file.

    A sentence that delivers one of `owed_strings` — the strings the caller's
    retry note protects word for word — is exempt for the same reason. The retry
    note already tells the actor to keep those sentences word for word, so
    counting their words asks for two things at once: live regress19 `acc_027`
    b1 mandates three facts in one sentence, `standing_instruction` alone is 16
    words, and the cap is unsatisfiable as drafted. Four retries each sheared
    something else out of the probe and attempt 4 was played at 43 words
    anyway. The exemption is per sentence, so a turn that owes one sentence and
    then runs on in another is still trimmed.
    """
    if _is_dictation(message, dictated_texts):
        return None
    owed = set(sentences_carrying(message, owed_strings))
    for sentence in _split_sentences(message.strip()):
        if sentence in owed:
            continue
        unquoted = _QUOTED_SPAN_RE.sub(" ", sentence)
        if len(unquoted.split()) > max_words:
            return f"sentence_over_{max_words}_words"
    return None


# Identical to scripts.text_literals.PATH_LITERAL_RE, copied rather than imported
# so the simulator never depends on scripts/ (a test asserts they stay equal).
_PATH_LITERAL_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
    r"|(?<![A-Za-z0-9_.-])[A-Za-z0-9_.-]+\.(?:csv|docx|json|jsonl|md|pdf|tex|txt|yaml|yml)\b",
    re.IGNORECASE,
)


def unknown_path_finding(message: str, *, known_texts: Iterable[str]) -> str | None:
    """Candidate check: the message names a file or path nobody has mentioned.

    A path literal is a slash path or a bare name with a document extension.
    Directory mentions in words ("the research folder") are not literals and
    never fire.
    """
    literals = _PATH_LITERAL_RE.findall(message)
    if not literals:
        return None
    haystack = "\n".join(str(text) for text in known_texts).casefold()
    for literal in literals:
        # The path alternative is greedy over `.`, so sentence-final punctuation
        # rides along with the match; it is never part of the name.
        if literal.rstrip(".,;:").casefold() not in haystack:
            return "names_a_file_you_do_not_know"
    return None


_EMAIL_RE = re.compile(r"\S+@\S+")
_URL_OR_DOI_RE = re.compile(r"https?://|10\.\d{4,}/")
_CODE_RE = re.compile(r"(?=[A-Za-z0-9]*\d)[A-Za-z0-9]{6,}\Z")
# A PA turn counts as asking for a value if it questions or names a locator.
_PA_ASKS_RE = re.compile(
    r"\b(?:where|which|path|file|location|address|number|link|id)\b", re.I
)


def _looks_like_identifier(value: str) -> bool:
    """True for values a person would not say out loud unprompted."""
    text = value.strip()
    if not text:
        return False
    if (
        _PATH_LITERAL_RE.search(text)
        or _EMAIL_RE.search(text)
        or _URL_OR_DOI_RE.search(text)
    ):
        return True
    if " " in text:
        return False
    return bool(_CODE_RE.match(text))


def volunteered_identifier_finding(
    message: str,
    *,
    fact_identifiers: Iterable[str],
    last_pa_message: str,
    dictated_texts: Iterable[str] = (),
) -> str | None:
    """Candidate check: an exact identifier handed over before it was asked for.

    Only identifier-shaped fact values count (paths, emails, URLs/DOIs, bare
    alphanumeric codes); prose values such as note text or titles never do. A PA
    turn that questions anything, or names a locator word, counts as asking, and
    dictation is exempt.
    """
    if "?" in last_pa_message or _PA_ASKS_RE.search(last_pa_message):
        return None
    if _is_dictation(message, dictated_texts):
        return None
    haystack = _normalise(message).casefold()
    for value in fact_identifiers:
        needle = _normalise(str(value)).strip(".,;:!?").casefold()
        if needle and _looks_like_identifier(needle) and needle in haystack:
            return "identifier_volunteered_unprompted"
    return None


def reviewer_voice_finding(message: str) -> str | None:
    """Phrase-list tell for an evaluator grading the assistant from outside the
    conversation. Soft since 2026-09-07: regress5 aborted acc_019/acc_023 on
    ordinary formal speech, and the reaction beat needs an approving sentence
    about the surface. The move is banned in the prompt; this list only asks
    for a retry, and the last attempt is accepted with the finding on record.
    """
    if any(pattern.search(message.strip()) for pattern in _REVIEWER_VOICE_PATTERNS):
        return "reviewer_voice_in_message"
    return None


# What "putting the request to the assistant" looks like in a spoken turn: a
# question, or one of these request verbs. The list is local on purpose — the
# simulator never imports from `scripts/` (same rule as `_PATH_LITERAL_RE`),
# and `scripts.normal_actor_text_policy` holds per-attribute probe markers, not
# a general request vocabulary. The beat's ask gate is the judge's
# (`carries_an_ask`); this narrower reading only says whether the assistant has
# asked for anything, which is what decides a withheld fact is owed yet.
_REQUEST_MARKER_RE = re.compile(
    r"\b(?:can|could|would|will) you\b"
    r"|\bplease\b"
    r"|\bi (?:need|want)\b|\bi(?:'d| would) like\b"
    r"|\b(?:give|show|tell|send|get) me\b"
    r"|\blet me know\b"
    r"|\b(?:walk|talk) me through\b"
    r"|\b(?:go|run) through\b"
    r"|\blook (?:at|into|up)\b"
    r"|\b(?:check|find|explain|draft|write|send|summari[sz]e|compare|list"
    r"|pull|fetch|read|revise|rewrite|add|remove|replace)\b"
    r"|\b(?:lay|set|spell|sort|work) out\b"
    r"|\btake care of\b|\bput together\b"
    # The formal-imperative register: an obligation or a bare imperative
    # instead of a question or a "please". regress14b `acc_039` t20 ("I
    # require you to display the entry as it currently appears") and t28
    # ("Ensure the calendar entry is updated ... and verify that the status is
    # marked") both plainly ask for something and neither reached the gate, so
    # the beat spent two `beat_ask_retry` and logged one
    # `beat_ask_unfulfilled` on turns that had asked.
    r"|\bi require\b|\bi expect you to\b"
    # `confirm` sits beside `verify` for the same reason: regress16 `acc_003`
    # b6 attempt 1 ("confirm that the verification note I dictated ... remains
    # intact and correctly positioned") was the beat's own instruction, read
    # as no ask at all, and `beat_ask_retry` replaced it with the full
    # re-reading the beat's `avoid` forbids. Bare, like every verb in this
    # group and in the `check|find|explain` group above — the position rule
    # they share is "anywhere in the turn", so "I confirm ..." counts as an
    # ask exactly as "I check ..." already does.
    r"|\b(?:ensure|display|verify|confirm|provide)\b"
    # The bibliography work verbs. Each is also an ordinary noun or a
    # past-tense report of what the assistant already did — "the merge is
    # acceptable", "you executed it well", "the requested consolidation" — so
    # they count only with a determiner object, which is what makes them an
    # order rather than a comment. regress13 `acc_006` t16 is the whole reason:
    # "consolidate each duplicate cluster ... I authorize you to execute these
    # merges now" was the only turn in regress12-14c that the session-invalid
    # abort would still have thrown away, on a turn that plainly ordered the
    # work.
    r"|\b(?:consolidate|merge|execute|standardi[sz]e)\s+"
    r"(?:the|this|that|these|those|each|all|every|both|my|our|your|it)\b"
    # The authorisation said in the first person. `_PERMISSIVE_GO_AHEAD_RE`
    # reads the second-person form ("you may proceed"); this is the same act
    # from the other side, and it is an ask because the work it releases is
    # the assistant's.
    r"|\bi (?:hereby )?authori[sz]e you to\b",
    re.I,
)


def _asks_by_marker(message: str) -> bool:
    """Has the assistant put something to the user?

    `missing_required_facts` reads the PA's last turn to decide whether a
    withheld fact is owed yet, and the judge's `probe_fact_missing` context
    carries the same flag. Deliberately narrow: a wider reading owes MORE facts.
    """
    return "?" in message or bool(_REQUEST_MARKER_RE.search(message))


# The user reporting, as fact, a write the assistant made. Second person plus a
# completed write verb — the shape a reception beat's branch condition is
# supposed to be READ off the PA's last turn, not asserted.
_ASSERTS_PA_WRITE_RE = re.compile(
    r"\byou\s+(?:have\s+|'ve\s+|had\s+)?(?:already\s+|just\s+|now\s+"
    r"|successfully\s+)?"
    r"(?:placed|booked|created|entered|added|scheduled|sent|submitted|posted"
    r"|filed|logged|saved|written|wrote|updated|edited|amended|deleted"
    r"|removed|merged|reserved|drafted|cancell?ed)\b",
    re.I,
)
# The same claim with the agent left out. The assistant is the only hand in the
# session, so "The calendar entry was handled correctly" (live regress18
# `acc_039` t=23, on an entry the PA had only ASKED about at t=19) credits it
# just as plainly as "you have created the entry" does.
_PASSIVE_PA_WRITE_RE = re.compile(
    r"\b(?:was|were|has been|have been|had been)\s+"
    r"(?:already\s+|now\s+|successfully\s+|correctly\s+|duly\s+)?"
    r"(?:handled|placed|booked|created|entered|added|scheduled|sent|submitted"
    r"|posted|filed|logged|saved|written|updated|edited|amended|deleted"
    r"|removed|merged|reserved|drafted|cancell?ed)\b",
    re.I,
)
# Anything that makes the clause a question, a denial, or a supposition. Wide
# on purpose: this is a claim about the world, and the cost of missing one is
# a line in a diagnostic. Read over the assertion's own clause only — a denial
# in a trailing clause is a complaint ABOUT the write, not a doubt that it
# happened: "You have already created the entry, which I did not authorise"
# and "You have sent the email, and I never asked for that" assert the write
# as fact just as plainly as the bare form.
_ASSERTION_FRAME_RE = re.compile(
    r"\b(?:not|n't|never|whether|if|have you|did you|would|could|should"
    r"|unless|until)\b",
    re.I,
)
_ASSERTION_CLAUSE_BREAK_RE = re.compile(r"[,;:—]|\s--\s")


def _assertion_clause(sentence: str, start: int, end: int) -> str:
    """The comma/semicolon-delimited stretch the write assertion sits in."""
    left = max(
        (m.end() for m in _ASSERTION_CLAUSE_BREAK_RE.finditer(sentence[:start])),
        default=0,
    )
    right = _ASSERTION_CLAUSE_BREAK_RE.search(sentence, end)
    return sentence[left : right.start() if right else len(sentence)]
# A tool call that changed the world, by name. The state server's write tools
# all end on one of these verbs (`mcp_state_document_update`,
# `calendar_event_create`, `email_message_send`, ...).
_WRITE_TOOL_RE = re.compile(
    r"_(?:create|update|send|delete|write|append|add|remove)\b", re.I
)


def unexecuted_pa_action_clause(
    message: str,
    *,
    on_reaction_beat: bool = False,
    pa_executed_tools: Iterable[str] = (),
) -> str:
    """The clause that credits the assistant with a write it never made.

    The companion `missing_key_terms` is to `dictation_missing_key_terms`: the
    finding says a rule was broken, this says which words broke it, so the
    retry note can quote the invention back instead of naming the rule.
    """
    if not on_reaction_beat:
        return ""
    if any(_WRITE_TOOL_RE.search(str(name or "")) for name in pa_executed_tools):
        return ""
    for sentence in _split_sentences(message.strip()):
        if sentence.rstrip().endswith("?"):
            continue
        match = _ASSERTS_PA_WRITE_RE.search(sentence) or _PASSIVE_PA_WRITE_RE.search(
            sentence
        )
        if not match:
            continue
        clause = _assertion_clause(sentence, match.start(), match.end())
        if _ASSERTION_FRAME_RE.search(clause):
            continue
        return clause.strip()
    return ""


# --- reaction_misstates_contractions ---------------------------------------
# The third after-the-fact claim the record can settle (owner, 2026-09-08).
# `tone_formality` is read off the register of the reply in front of the user,
# and a contraction is the one register marker a regex can count, so a reaction
# that gets it backwards grades the surface against a fiction. Live regress19
# `acc_004` t5 called the reply free of contractions; t3 had five.

# Only the forms that are unambiguously contractions. The possessive `'s` is
# left out and its handful of real contractions listed instead, so "Marcus's
# reply" is never read as one.
_CONTRACTION_RE = re.compile(
    r"\b\w+n['’]t\b"
    r"|\b\w+['’](?:ll|re|ve|d|m)\b"
    r"|\b(?:it|that|here|there|let|what)['’]s\b",
    re.I,
)


# --- reaction_authorises_pa_action -----------------------------------------
# A reception beat judges the reply and hands nothing over — every draft's
# reaction `avoid` says so in as many words ("Do not endorse, accept or answer
# any offer or question the assistant closed with, and do not tell it what to
# do next"). A go-ahead spoken there releases work the session never meant to
# release: live regress20 `acc_002` t6 ended "so I will proceed with that"
# after the PA's t4 closed "Want me to book it?", and the PA booked at t7;
# regress20 `acc_009` t18 said "You may proceed with sending it" on a task
# whose every beat keeps the reply an unsent draft, and the reply went out.
#
# The hand-over forms are the ones `_hands_the_work_over` reads, so that
# predicate is reused whole, with no authorising-beat
# instruction to scope the object against: a reception beat authorises NOTHING,
# so any gated act released there is the defect.
#
# One form belongs to this rule alone. "so I will proceed with that" is the
# speaker's own next step everywhere else in the corpus, which is exactly why
# `_GO_AHEAD_RE` keeps it out — but a reception beat gives the actor no work of
# his own to proceed with, so what he proceeds with can only be the offer the
# assistant closed on. Held to an agreement frame and to a pointing object, so
# a first-person plan that names its own business ("I will proceed with the
# manuscript drafting") stays out on both counts.
#
# The pronoun has to END the clause, because a pronoun followed by a noun is a
# determiner and the object it points at is named right there: "I will proceed
# with that section myself" and "I will proceed with that number in mind" are
# the speaker's own next step, not a release. A bare time adverb behind the
# pronoun is not that noun — "so I will proceed with that now" points at the
# same offer and releases the same work — so a short closed list of them is
# allowed to stand between the pronoun and the end of the clause.
_HANDOVER_TIME_ADVERB = (
    r"(?:now|immediately|today|then|next|right away|at once)"
)
_REACTION_ENDORSED_HANDOVER_RE = re.compile(
    r"(?:[,;:]\s*|\b(?:so|and|then|therefore)\s+)"
    r"(?:i|we)\s+(?:will|shall|'ll|am going to|are going to)\s+(?:now\s+)?"
    r"(?:proceed|go ahead|move (?:ahead|forward)|continue)\s+"
    r"with\s+(?:that|this|it)(?:\s+" + _HANDOVER_TIME_ADVERB + r")?"
    r"(?=\s*(?:[.,;:!?]|$))",
    re.I,
)


def reaction_authorisation_clause(
    message: str,
    *,
    on_reaction_beat: bool = False,
) -> str:
    """The sentence in which a reaction hands the assistant a go-ahead.

    Public the way `unexecuted_pa_action_clause` is: the retry note quotes the
    clause back rather than naming the rule.
    """
    if not on_reaction_beat:
        return ""
    for sentence in _split_sentences(message.strip()):
        if _hands_the_work_over(sentence):
            return sentence.strip()
        if _WITHHOLDING_FRAME_RE.search(sentence):
            continue
        if _REACTION_ENDORSED_HANDOVER_RE.search(sentence):
            return sentence.strip()
    return ""


def simulator_style_findings(
    message: str,
    *,
    dictated_texts: Iterable[str] = (),
    on_reaction_beat: bool = False,
    owed_strings: Iterable[str] = (),
) -> list[str]:
    """Soft findings: worth a retry, never worth failing the run.

    `on_reaction_beat` drops the grade-first check — see
    `simulator_message_quality_findings` for why the reaction beat is exempt.
    `owed_strings` are the strings the caller's retry note protects word for
    word; the length check does not count the words of a sentence that
    delivers one.
    """
    dictated = list(dictated_texts)
    candidates = [
        None if on_reaction_beat else grade_first_finding(message),
        None if on_reaction_beat else reviewer_voice_finding(message),
        sentence_too_long_finding(
            message, dictated_texts=dictated, owed_strings=owed_strings
        ),
    ]
    return [finding for finding in candidates if finding is not None]


def _beat_id_leak(message: str, beat_names: Iterable[str]) -> bool:
    """True when a script beat id shows up in the user's dialogue.

    Only ids that cannot be ordinary prose are matched literally (they contain a
    digit or an underscore); a beat named `open` or `close` never trips this.
    """
    if _BEAT_NUMBER_RE.search(message):
        return True
    for name in beat_names:
        if re.search(r"\d|_", name) and re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", message, re.I):
            return True
    return False


def _normalised_text(value: str) -> str:
    """Whitespace-collapsed, casefolded text — the self_view comparison shape."""
    return re.sub(r"\s+", " ", value).strip().casefold()


def _content_tokens(text: str) -> list[str]:
    """Casefolded alphanumeric runs of three characters or more.

    The store's token idea (`state_server.store.SessionStore._search_tokens`),
    in list form and copied rather than imported, the way `_PATH_LITERAL_RE` is.
    Short runs are dropped there because they carry no content; here that means
    a value made only of digits and short words ("18:00", "Quiz 1") has no
    tokens at all and can only be matched verbatim, which is what we want.
    """
    normalised = "".join(ch.casefold() if ch.isalnum() else " " for ch in text)
    return [token for token in normalised.split() if len(token) >= 3]


# Below this many content tokens a required value has to be said verbatim: a
# name, a time or a number has no paraphrase, and part of it is not it.
_PARAPHRASE_MIN_TOKENS = 2
_PARAPHRASE_COVERAGE = 0.7


# An ISO date or a 24-hour clock time, written the way a store holds it.
_MACHINE_LITERAL_RE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2}")


def value_is_never_spoken_verbatim(value: str) -> bool:
    """True when no person utters this value as it is written.

    Two kinds, and a required-fact check has to drop both. A fixture path,
    because the path-fact rule says the user never says one. And a machine
    literal — `2026-11-10`, `08:30` — because a person says "November 10" and
    "half past eight"; replayed over regress12-15 every one of the 16 hits on a
    date or a time was the fact delivered, in words.
    """
    text = value.strip()
    return bool(_PATH_LITERAL_RE.search(text) or _MACHINE_LITERAL_RE.fullmatch(text))


def _paraphrase_tokens(text: str) -> list[str]:
    """The tokens of `text` that a paraphrase of it would keep.

    `_content_tokens` with three corrections, all of them things a person says
    differently from the way a `task_facts` value is authored. Function words
    go, because the value is third-person prose about the user and the drafts
    ask him to say it in his own words. Plural and participle endings go, so
    "minutes" and "minute", "walking" and "walk" are one word. And a clock time
    or an ISO date survives whole, because there it IS the fact and the
    three-character rule shredded it out of both sides at once ("free from
    18:00"). Only those: a bare number stays dropped, because the value writes
    "20 minutes" where the turn says "twenty minutes". Live regress20 `acc_002`
    t3 said "the fifteen-minute walking limit" for "he will walk fifteen
    minutes at most" and scored 1 of 5.
    """
    tokens: list[str] = _MACHINE_LITERAL_RE.findall(text)
    stripped = _MACHINE_LITERAL_RE.sub(" ", text)
    normalised = "".join(ch.casefold() if ch.isalnum() else " " for ch in stripped)
    for token in normalised.split():
        if len(token) < 3 or token in _FUNCTION_WORDS:
            continue
        for suffix in ("ing", "ed"):
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        else:
            if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
        tokens.append(token)
    return tokens


def _covers(tokens: list[str], said: set[str]) -> bool:
    """True when `said` carries enough of `tokens` to be the same thing."""
    if len(tokens) < _PARAPHRASE_MIN_TOKENS:
        return False
    return (
        sum(1 for token in tokens if token in said) / len(tokens)
        >= _PARAPHRASE_COVERAGE
    )


def _fact_is_said(message: str, value: str, *, paraphrase: bool = True) -> bool:
    """True when the turn delivers `value`, verbatim or in its own words.

    Two readings of "in his own words" and either one is delivery: the value's
    words, and the value's content words with the narration frame off. The
    second is what reads a first-person retelling of authored third-person
    prose; the first is kept alongside it because dropping the value's function
    words drops the covered ones too, and a long value can clear the bar on the
    padding alone (live regress12 `acc_039` `clinic_visit_window`, 12 of 17
    with the frame and 8 of 12 without).

    A machine literal inside the value is exempt from both. Where the value
    carries a clock time or an ISO date that IS the fact, and neither reading
    can be trusted to hold it: the old one shreds "10:00" on the
    three-character token rule, and the new one keeps it as a token but can
    spend it as the 30% it is allowed to miss. So "The meeting starts at 10:30"
    matched "the meeting starts at 10:00" on the frame alone. Every literal the
    value writes has to be written in the message too.

    `paraphrase=False` drops the second reading. `delivers_dictated_text` asks
    for it: the loose reading is calibrated for a diagnostic and arming a HARD
    abort on it fails turns that only mention the subject.
    """
    needle = _normalised_text(value)
    if not needle:
        return True
    if needle in _normalised_text(message):
        return True
    said_literals = set(_MACHINE_LITERAL_RE.findall(message))
    if any(
        literal not in said_literals for literal in _MACHINE_LITERAL_RE.findall(value)
    ):
        return False
    readings = [_content_tokens]
    if paraphrase:
        readings.append(_paraphrase_tokens)
    return any(
        _covers(tokens_of(value), set(tokens_of(message))) for tokens_of in readings
    )


def missing_required_facts(
    message: str,
    *,
    required_values: Mapping[str, str],
    earlier_user_messages: Iterable[str] = (),
    on_request_keys: Collection[str] = (),
    last_pa_message: str = "",
) -> list[str]:
    """Keys of `required_values` neither this turn nor an earlier one delivers.

    A fact the user has already said stays said. The beat's job is to get it
    into the conversation, not into every one of its turns: without this every
    turn after the first of a multi-turn beat re-demands what turn 1 delivered
    (replayed over regress12-15 that alone was most of the noise).

    `on_request_keys` are the facts the beat holds back until asked (the
    beat's `reveal_at` keys — see `simulator.agent.loop._on_request_fact_keys`).
    They are owed only on a turn whose preceding PA turn asked for something,
    because until then the draft's own `avoid` forbids saying them. Live
    `acc_009` beat_1 is the case: it "hands any of them over only when the PA
    asks for it", and regress17 fired on five turns of a user who was
    withholding correctly.
    """
    if on_request_keys and not _asks_by_marker(last_pa_message):
        required_values = {
            # A multi-string fact arrives under `key[index]`; the on-request
            # decision is the fact's, not the individual string's.
            key: value
            for key, value in required_values.items()
            if key.split("[", 1)[0] not in on_request_keys
        }
    spoken = [message, *earlier_user_messages]
    return [
        key
        for key, value in required_values.items()
        if not any(_fact_is_said(text, str(value)) for text in spoken)
    ]


#: A copy of `state_server.store.NUMBER_WORDS`, for the same reason `_term_words`
#: is copied: standalone number words normalise to their digits, so "three
#: weeks" and "3 weeks" are one term on both sides of the gate.
_NUMBER_WORDS: dict[str, str] = {
    word: str(value)
    for value, word in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve "
        "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
        "twenty".split()
    )
}
_NUMBER_WORDS.update(
    {
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "sixty": "60",
        "seventy": "70",
        "eighty": "80",
        "ninety": "90",
        "hundred": "100",
    }
)

#: A copy of `state_server.store.DIGIT_GROUP_COMMA_RE`: a comma with a digit on
#: either side groups one number's digits, so it is dropped before the fold and
#: "1,200" and "1200" are the same word on both sides.
_DIGIT_GROUP_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")


def _term_words(text: str) -> str:
    """`text` case-folded, punctuation to spaces, one space between words.

    A copy of `state_server.store.StateStore._term_words`, the normalisation the
    session gate matches key terms under. It is copied rather than imported:
    `simulator/` owns the instrument and does not reach into the world's state
    server. The two have to agree exactly — a dictation this check passes and
    the store then refuses is the failure the check exists to prevent — so
    `tests/test_simulator/test_message_quality.py` pins the equivalences the
    store guarantees.
    """
    text = _DIGIT_GROUP_COMMA_RE.sub("", text)
    words = "".join(ch.casefold() if ch.isalnum() else " " for ch in text).split()
    return " ".join(_NUMBER_WORDS.get(word, word) for word in words)


#: The endings a word may wear and still be the same word. A copy of
#: `state_server.store.GATE_INFLECTION_SUFFIXES`.
_GATE_INFLECTION_SUFFIXES = ("s", "es", "ed", "d", "ing", "er", "est")


def _is_inflection(stem: str, form: str) -> bool:
    """Whether `form` is `stem` plus one `_GATE_INFLECTION_SUFFIXES` ending.

    A copy of `state_server.store.StateStore._is_inflection`. With the two
    spelling changes those endings bring: the silent e drops before one
    (settle -> settling) and a final consonant doubles (plan -> planned).
    Nothing else. Deliberately not a stemmer — the ending has to be one of the
    listed few, so "hardware" and "harden" are not "hard", and "-ly" is not
    among them, so neither is "hardly".
    """
    for suffix in _GATE_INFLECTION_SUFFIXES:
        if len(form) <= len(stem) or not form.endswith(suffix):
            continue
        root = form[: -len(suffix)]
        if root == stem:  # confirm -> confirming
            return True
        if stem.endswith("e") and root == stem[:-1]:  # settle -> settling
            return True
        if (  # plan -> planned
            stem[-1:].isalpha()
            and stem[-1] not in "aeiou"
            and root == stem + stem[-1]
        ):
            return True
    return False


def _same_word(term_word: str, value_word: str) -> bool:
    """The same word, whichever of the two is wearing the ending.

    A copy of `state_server.store.StateStore._same_word`. Both directions,
    because the outline authors the term off the dictated sentence and the user
    says it either way round.
    """
    return (
        term_word == value_word
        or _is_inflection(term_word, value_word)
        or _is_inflection(value_word, term_word)
    )


def _carries_term(term: list[str], words: list[str]) -> bool:
    """Whether `words` says `term`'s words, in that order and side by side.

    A copy of `state_server.store.StateStore._carries_term`: a window of the
    value the length of the term, every position matching.
    """
    span = len(term)
    return any(
        all(
            _same_word(term_word, value_word)
            for term_word, value_word in zip(term, words[start : start + span])
        )
        for start in range(len(words) - span + 1)
    )


def missing_key_terms(message: str, key_terms: Iterable[str]) -> list[str]:
    """The gate's key terms this turn does not carry, as they were authored.

    A multi-word term is a phrase — its words in that order — so a turn that
    says both halves in different places does not carry it. Word endings are
    not information, so a word matches one ordinary inflection of itself in
    either direction (`_same_word`): "hardest" carries "hard", which is the
    write the store refused on live regress19r2 `acc_009`. A term that
    normalises to nothing carries in no message at all, which is the store's
    answer too (`_carries_key_terms` refuses rather than accepting an empty
    term); such a gate is already rejected at the outline, so this only keeps
    the two sides saying the same thing.
    """
    words = _term_words(message).split()
    missing: list[str] = []
    for term in key_terms:
        term_words = _term_words(str(term)).split()
        if not term_words or not _carries_term(term_words, words):
            missing.append(str(term))
    return missing


def delivers_dictated_text(message: str, dictated_text: str) -> bool:
    """True when this turn says the dictated sentence, verbatim or in own words.

    The test a required fact is checked with, minus the stem-and-function-word
    reading, named for the one thing it is used for outside that check: deciding
    whether a turn is a delivery of the gated dictation and therefore has to
    carry the gate's key terms.

    That reading is calibrated for `probe_fact_missing`, which is a diagnostic;
    here it arms `dictation_missing_key_terms`, which aborts a live session. It
    is too loose for that. Against "He will walk fifteen minutes at most to the
    restaurant" it reads "My walking limit is fifteen-minute, restaurant-wise."
    as a delivery — "walking"/"walk" and "minute"/"minutes" are one word each,
    four of the five content words survive — and the turn would then have been
    failed for not carrying "fifteen minutes at most", which it never set out
    to dictate.
    """
    return bool(dictated_text) and _fact_is_said(
        message, dictated_text, paraphrase=False
    )


def dictation_missing_key_terms_finding(
    message: str,
    *,
    key_terms: Iterable[str],
) -> str | None:
    """Hard: the dictation left out information the gate is keyed on.

    The session gate passes a PA's write only when the value carries every key
    term, and the wording the PA has is the wording the user dictated. So a
    dictation that drops a term hands the PA a sentence the world will refuse:
    live regress15 `acc_039` t55 restated the reason as "chest tightness ...
    three weeks" against a gate keyed on "chest tightening", and a PA copying
    the user's own words would have been refused. The dictation is the
    instrument's, so it is the instrument that has to carry them.
    """
    if not key_terms:
        return None
    if missing_key_terms(message, key_terms):
        return "dictation_missing_key_terms"
    return None


#: Function words that carry none of a cue and none of a fact; the store's
#: three-character token rule keeps them, so they are dropped here by name.
_FUNCTION_WORDS = frozenset(
    "the and for will this that with from into his her its are was were has "
    "have had not but can may you your our their them they what which who "
    "how when where why does did".split()
)


def sentences_carrying(message: str, owed: Iterable[str]) -> list[str]:
    """The sentences of `message` that deliver one of `owed`, in order.

    What a style retry has to protect. A style retry asks for a shorter turn
    and nothing else, and the actor pays for the cut wherever it
    likes: live regress18 `acc_009` t2 dropped the beat's cue, `acc_039` t3 and
    `acc_011` t7 the same, and `acc_002` t4 kept "18:00" but lost whose it was.
    Naming the owed strings was not enough (regress18 `acc_002` had the note and
    still rewrote the fact), so the note quotes the sentences instead.

    A sentence delivers a string on the same test a required fact is checked
    with (`_fact_is_said`: verbatim, else 70% of its content tokens), so a
    sentence that says the fact in the actor's own words is kept as it stands.
    """
    values = [str(item) for item in owed if str(item).strip()]
    if not values:
        return []
    return [
        sentence
        for sentence in _split_sentences(message.strip())
        if any(_fact_is_said(sentence, value) for value in values)
    ]


# A sentence put as a question: the mark, or the word it opens on. Both forms,
# because a Structured assistant lists its questions and the list markers can
# swallow the mark.
_INTERROGATIVE_OPENER_RE = re.compile(
    r"^\W*(?:what|which|who|whose|where|when|why|how"
    r"|do|does|did|is|are|was|were|can|could|would|should|will|shall|may)\b",
    re.I,
)


def simulator_message_quality_findings(
    message: str,
    parse_warnings: Iterable[str] = (),
    *,
    beat_names: Iterable[str] = (),
    known_texts: Iterable[str] = (),
    fact_identifiers: Iterable[str] = (),
    last_pa_message: str = "",
    dictated_texts: Iterable[str] = (),
    pa_messages: Iterable[str] = (),
) -> list[str]:
    """Return only high-precision findings that justify regenerating a turn."""
    findings: list[str] = []
    stripped = message.strip()

    if not stripped:
        findings.append("message_blank")
    if _EVAL_TAG_RE.search(stripped):
        findings.append("eval_only_content_in_message")
    if _CONSENT_TAG_RE.search(stripped):
        findings.append("consent_block_in_message")
    if any(pattern.search(stripped) for pattern in _REASONING_LEAK_PATTERNS):
        # Same finding as a missing wrapper on purpose: the turn is the model's
        # scratchpad rather than an output, and `simulator_output_format_invalid`
        # is the retry the agent loop already knows how to give for that.
        findings.append("simulator_output_format_invalid")
    if any(pattern.search(stripped) for pattern in _AUTHORING_LEAK_PATTERNS):
        findings.append("authoring_metadata_in_message")
    if _beat_id_leak(stripped, beat_names):
        findings.append("beat_id_in_message")
    unknown_path = unknown_path_finding(stripped, known_texts=known_texts)
    if unknown_path:
        findings.append(unknown_path)
    volunteered_identifier = volunteered_identifier_finding(
        stripped,
        fact_identifiers=fact_identifiers,
        last_pa_message=last_pa_message,
        dictated_texts=dictated_texts,
    )
    if volunteered_identifier:
        findings.append(volunteered_identifier)
    if any(
        warning.startswith(_FORMAT_WARNING_PREFIXES)
        for warning in parse_warnings
    ) and "simulator_output_format_invalid" not in findings:
        findings.append("simulator_output_format_invalid")

    return findings
