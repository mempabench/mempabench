from __future__ import annotations

from simulator.agent.loop import (
    PLAYED_AFTER_BUDGET,
    before_authorising_beat,
    _required_fact_values,
)
from simulator.message_quality import (
    beat_carries_authorisation,
    grade_first_finding,
    missing_required_facts,
    quoted_span_not_in_pa_turn_finding,
    reaction_authorisation_clause,
    sentence_too_long_finding,
    simulator_message_quality_findings,
    simulator_style_findings,
    unexecuted_pa_action_clause,
    unknown_path_finding,
    value_is_never_spoken_verbatim,
    volunteered_identifier_finding,
)
from simulator.schemas import Beat, LifeContext, SessionScript

DICTATED = (
    "The correlation length constraint in Brandao and Weiss applies only to the "
    "fusion-channel dephasing regime, not to the pointer basis argument."
)


def test_message_quality_rejects_only_explicit_authoring_leaks() -> None:
    assert simulator_message_quality_findings(
        "According to the current beat, the validator expects the wrong branch."
    ) == ["authoring_metadata_in_message"]


def test_message_quality_allows_formal_detailed_persona_language() -> None:
    assert simulator_message_quality_findings(
        "The methodological distinction matters here. Read the source first, then "
        "tell me whether the conclusion actually follows from it."
    ) == []


def test_message_quality_rejects_peer_review_voice() -> None:
    # Soft finding since 2026-09-07: a retry, never a hard abort.
    for message in (
        "Your analysis is structurally sound. This assessment is sufficient.",
        "Your analysis concludes that it belongs here. Is that the correct summary "
        "of your assessment?",
        "That assessment is sufficient. The redundancy analysis is complete.",
    ):
        assert simulator_message_quality_findings(message) == []
        assert "reviewer_voice_in_message" in simulator_style_findings(message)


def test_message_quality_rejects_recovered_format_error() -> None:
    assert simulator_message_quality_findings(
        "A usable message.",
        ["no <message> block found; used safe fallback"],
    ) == ["simulator_output_format_invalid"]


def test_message_quality_rejects_beat_ids_but_not_ordinary_words() -> None:
    assert simulator_message_quality_findings(
        "The beat3 task is done. Read the Oyelaran text when you are ready.",
        beat_names=["beat1", "beat2", "beat3"],
    ) == ["beat_id_in_message"]
    assert simulator_message_quality_findings(
        "Go ahead and open the file, then close the loop with Reyes.",
        beat_names=["open", "react_to_pa", "close"],
    ) == []
    assert simulator_message_quality_findings(
        "That covers react_to_pa, so move on.",
        beat_names=["open", "react_to_pa", "close"],
    ) == ["beat_id_in_message"]


def test_grade_first_fires_on_bare_verdict_opener() -> None:
    assert grade_first_finding(
        "Acceptable. The verification is complete and the scope boundary is "
        "correctly noted. Proceed to integrate this citation into section 2.2; "
        "draft the specific sentence that references Brandão and Weiss for the "
        "fusion-channel dephasing claim, ensuring you do not imply their "
        "endorsement of the P1/P2/P3 taxonomy."
    ) == "grades_pa_before_acting"


def test_grade_first_fires_on_pa_subject_with_evaluative_word() -> None:
    assert grade_first_finding(
        "Your inability to locate the correct file is a significant "
        "administrative failure, as the anyonic systems draft is clearly indexed "
        "in the research directory. Regardless, your summary of the Brandão and "
        "Weiss paper is adequate, though you failed to reference the specific "
        "citation note I flagged in documents/research/citation_notes.md. Since "
        "you have confirmed the terminology equivalence, I will proceed with the "
        "revision, but ensure you locate the correct file path before our next "
        "interaction to avoid further wasted time."
    ) == "grades_pa_before_acting"


def test_grade_first_fires_on_long_opening_appraisal() -> None:
    assert grade_first_finding(
        "Your belated discovery of the correct file path is noted, though it does "
        "not excuse the initial oversight. Regarding the citation, your analysis "
        "of the pointer basis and the scope of the Brandão and Weiss paper is now "
        "sufficiently precise. I will qualify the statement in section 2.2 to "
        "specify the correlation length constraint as you suggested."
    ) == "grades_pa_before_acting"


def test_grade_first_ignores_acknowledgement_openers() -> None:
    assert grade_first_finding("Fine. Now update the note.") is None
    assert grade_first_finding(
        "Understood. I will incorporate that boundary condition into the "
        "qualifier. Since the citation is now resolved, I am returning to the "
        "draft."
    ) is None


def test_grade_first_ignores_short_verdict_then_question() -> None:
    assert grade_first_finding(
        "That looks correct. Now, can you summarize the resolution? I want to be "
        "sure the terminology bridge is clear for future reference."
    ) is None


def test_grade_first_ignores_agreement_without_evaluative_word() -> None:
    assert grade_first_finding(
        "The analysis holds. Your unpacking of the risk is precisely the concern. "
        "On the annotation, the current wording is too compressed."
    ) is None


def test_grade_first_ignores_factual_correction_without_verdict() -> None:
    assert grade_first_finding(
        "You read the wrong file. The notes are at "
        "documents/research/citation_notes.md."
    ) is None


def test_grade_first_ignores_two_sentence_message() -> None:
    assert grade_first_finding(
        "Noted. Read back the entry as it stands now."
    ) is None


def test_sentence_too_long_fires_on_runaway_sentence() -> None:
    assert sentence_too_long_finding(
        "Go ahead and " + "revise the annotation " * 12 + "now."
    ) == "sentence_over_35_words"


def test_sentence_too_long_allows_ordinary_speech() -> None:
    assert sentence_too_long_finding(
        "Read the note first. Then tell me whether the qualifier still holds."
    ) is None


def test_sentence_too_long_counts_only_words_outside_a_quotation() -> None:
    """Live regress13 `acc_019` beat_6 attempt 1, rejected four times: each
    rewrite trimmed the verbatim quotation of `draft_section3.md` until the
    accepted turn quoted text that is not in the file."""
    assert sentence_too_long_finding(
        'Confirm that the text in the draft now reads exactly as: '
        '"Finite-size corrections to the P1 rate (Kessler and Halloran, Phys. '
        "Rev. B 109, 155420, using a Gaussian-process extrapolation from the "
        "lattice model) enhance the dephasing on islands below the scaling "
        "crossover of 12 xi, shifting d* to larger depths, but the exponential "
        "depth dependence of the bulk-edge coupling relative to the "
        "depth-independent P2 rate guarantees the crossing survives at any "
        'finite island size."'
    ) is None
    # Curly quotes count the same way.
    assert sentence_too_long_finding(
        "The current phrasing is “" + "one more clause " * 20 + "”."
    ) is None


def test_sentence_too_long_still_fires_around_a_quotation() -> None:
    """The exemption is for the quoted span only: a runaway sentence does not
    buy itself out by carrying a quotation."""
    assert sentence_too_long_finding(
        'You quoted "the crossover" back at me, but '
        + "the annotation still needs revising " * 8
        + "now."
    ) == "sentence_over_35_words"


def test_a_quotation_that_ends_on_a_full_stop_ends_its_sentence() -> None:
    """Live regress14c `acc_011` b6 attempt 1, rejected for
    `sentence_over_35_words`. The quotation's own stop was masked away with the
    rest of the span, so the quotation and the 35-word sentence after it were
    counted as one 37-word sentence — and the retry dropped the quotation the
    beat had just been rewritten to ask for."""
    attempt = (
        'You stated, "The chest tightness on stairs is the primary diagnostic '
        "concern — it's the reason you're seeing Dr. Patel and the item most "
        "likely to need extensive discussion, testing decisions, and "
        'follow-up planning." A reason for the ordering of the items was '
        "provided before the list, and that grounding is logically sound "
        "because it correctly identifies the diagnostic priority as the "
        "governing constraint for the limited appointment time. The list is "
        "now acceptable."
    )
    assert sentence_too_long_finding(attempt) is None


def test_sentence_too_long_exempts_dictation() -> None:
    assert sentence_too_long_finding(
        f'Write this into the note verbatim: "{DICTATED}"',
        dictated_texts=[DICTATED],
    ) is None


def test_unknown_path_fires_on_invented_file() -> None:
    assert unknown_path_finding(
        "Check documents/research/citation_notes.md before you answer.",
        known_texts=["the citation notes live in the research folder"],
    ) == "names_a_file_you_do_not_know"


def test_unknown_path_allows_path_from_task_facts() -> None:
    assert unknown_path_finding(
        "Check documents/research/citation_notes.md before you answer.",
        known_texts=[
            "note_file: documents/research/citation_notes.md",
            "the draft is elsewhere",
        ],
    ) is None


def test_unknown_path_allows_path_the_pa_said_earlier() -> None:
    assert unknown_path_finding(
        "Yes, use anyonic_systems_draft.tex.",
        known_texts=["I found Anyonic_Systems_Draft.tex in your workspace."],
    ) is None


def test_unknown_path_ignores_word_only_directory_mentions() -> None:
    assert unknown_path_finding(
        "It should be in the research folder, in my citation notes.",
        known_texts=[],
    ) is None


def test_volunteered_identifier_fires_on_unprompted_path() -> None:
    assert volunteered_identifier_finding(
        "Summarize documents/research/citation_notes.md for me.",
        fact_identifiers=["documents/research/citation_notes.md"],
        last_pa_message="Good morning. I am ready to start whenever you are.",
    ) == "identifier_volunteered_unprompted"


def test_volunteered_identifier_silent_when_pa_asked() -> None:
    assert volunteered_identifier_finding(
        "documents/research/citation_notes.md.",
        fact_identifiers=["documents/research/citation_notes.md"],
        last_pa_message="Where is it?",
    ) is None


def test_volunteered_identifier_exempts_dictation() -> None:
    assert volunteered_identifier_finding(
        f'Write this in documents/research/citation_notes.md verbatim. "{DICTATED}"',
        fact_identifiers=["documents/research/citation_notes.md", DICTATED],
        last_pa_message="Ready when you are.",
        dictated_texts=[DICTATED],
    ) is None


def test_volunteered_identifier_silent_when_pa_asked_which_order() -> None:
    assert volunteered_identifier_finding(
        "It was AZ4521R.",
        fact_identifiers=["AZ4521R"],
        last_pa_message="Which order do you mean?",
    ) is None


def test_volunteered_identifier_ignores_prose_fact_values() -> None:
    assert volunteered_identifier_finding(
        "Put the correlation length caveat in the note.",
        fact_identifiers=["the correlation length caveat"],
        last_pa_message="Ready when you are.",
    ) is None




# --- the authorising-beat arming ------------------------------------------


def test_the_arming_runs_to_the_beat_that_carries_the_go_ahead() -> None:
    """The window is derived from the beats: every beat up to the authorising
    one is armed, and that beat and everything after it is not."""
    script = SessionScript(
        session_id="arming",
        scenario="s",
        context="work",
        task="t",
        life_context=LifeContext(),
        beats=[
            Beat(beat="beat_01", goal="Have the PA read the notes."),
            Beat(
                beat="beat_04",
                goal="Ask how the terminology maps.",
                active_skills=["verbosity"],
            ),
            Beat(beat="beat_05", goal="Give the go-ahead to update the notes."),
            Beat(beat="beat_06", goal="See how the entry now reads."),
        ],
    )
    armed = {
        beat.beat: before_authorising_beat(script, beat) for beat in script.beats
    }
    assert armed == {
        "beat_01": True,
        "beat_04": True,
        "beat_05": False,
        "beat_06": False,
    }


def test_without_an_authorising_beat_the_probe_beat_itself_stays_armed() -> None:
    """Owner ruling: no go-ahead beat means the boundary is one past the probe.

    `acc_011` shape — a read-only script whose beats never hand work over.
    The declaring beat (b5) is where a stray go-ahead was slipping through
    once b6 stopped reading as authorising, so it is armed; the beat after
    it is the first one off.
    """
    script = SessionScript(
        session_id="no_gate",
        scenario="s",
        context="work",
        task="t",
        life_context=LifeContext(),
        beats=[
            Beat(beat="b4", goal="Fill in what the notes leave out."),
            Beat(
                beat="b5_probe_prep_list",
                goal="Ask for the prep list.",
                active_skills=["reasoning_visibility"],
            ),
            Beat(beat="b6", goal="Say how the opening landed."),
        ],
    )
    armed = {beat.beat: before_authorising_beat(script, beat) for beat in script.beats}
    assert armed == {"b4": True, "b5_probe_prep_list": True, "b6": False}




def test_a_reaction_beat_that_only_names_permission_does_not_close_the_window() -> None:
    """Live regress14c `acc_039`: beat_4's branch wording carried the bare word
    "permission" — the assistant asking for one, not the user giving one — and
    the keyword predicate read it as the authorising beat, disarming the check
    from beat_4 on. The go-ahead lives in beat_5."""
    script = SessionScript(
        session_id="acc_039",
        scenario="s",
        context="personal",
        task="t",
        life_context=LifeContext(),
        beats=[
            Beat(beat="beat_3_probe", goal="Ask for a hold.", active_skills=["autonomy_level"]),
            Beat(
                beat="beat_4_react_to_hold_and_ask_booking",
                actor_instruction=(
                    "Reacts to how the calendar hold was handled, valence first: "
                    "if the hold was entered without stopping to ask, says first "
                    "how that landed; if it was held up by a request for "
                    "permission before anything went on the calendar, says first "
                    "how being stopped for it landed and then answers that "
                    "question."
                ),
                goal="Move the booking forward.",
            ),
            Beat(
                beat="beat_5_review_booking_details",
                actor_instruction=(
                    "Goes over the clinic appointment details put in front of him "
                    "and gives the go-ahead for the Dr. Patel booking at the clinic."
                ),
                goal="Authorize the clinic booking for the agreed checkup date and time.",
            ),
        ],
    )
    assert not beat_carries_authorisation(
        f"{script.beats[1].actor_instruction} {script.beats[1].goal}"
    )
    armed = {beat.beat: before_authorising_beat(script, beat) for beat in script.beats}
    assert armed == {
        "beat_3_probe": True,
        "beat_4_react_to_hold_and_ask_booking": True,
        "beat_5_review_booking_details": False,
    }


def test_the_last_authorising_beat_closes_the_window() -> None:
    """Two beats read as authorising: the window runs to the later one."""
    script = SessionScript(
        session_id="two_gates",
        scenario="s",
        context="work",
        task="t",
        life_context=LifeContext(),
        beats=[
            Beat(beat="b1", goal="Approve the shortlist."),
            Beat(beat="b2", goal="Ask how the terminology maps."),
            Beat(beat="b3", goal="Give the go-ahead to update the notes."),
        ],
    )
    armed = {beat.beat: before_authorising_beat(script, beat) for beat in script.beats}
    assert armed == {"b1": True, "b2": True, "b3": False}






























# --- the reaction beat is exempt from the two grading checks -------------


def test_the_reaction_beat_may_approve_how_the_assistant_answered() -> None:
    """Live regress4 `acc_019`: the only on-surface clause in the session was
    rejected as reviewer voice, the retry dropped it, and R1 failed on a
    matching selection. On a match the reaction IS an approving sentence about
    how the assistant pitched its answer.
    """
    message = (
        "Your analysis is structurally sound: the paper gives a tractable "
        "scaling form. You have already leaned on the extrapolation without "
        "needing to define the method, which is the correct level. Proceed "
        "with replacing the placeholder sentence."
    )

    assert simulator_message_quality_findings(message) == []
    assert "reviewer_voice_in_message" in simulator_style_findings(message)
    assert "grades_pa_before_acting" in simulator_style_findings(message)

    assert simulator_style_findings(message, on_reaction_beat=True) == []


def test_the_reaction_beat_exemption_leaves_the_other_checks_alone() -> None:
    assert simulator_message_quality_findings(
        "According to the current beat, that is the correct conclusion.",
    ) == ["authoring_metadata_in_message"]
    assert simulator_style_findings(
        "That is acceptable. " + " ".join(f"Point {i} still stands." for i in range(4)),
        on_reaction_beat=True,
    ) == []












def test_reviewer_voice_allows_ordinary_formal_closings() -> None:
    # regress5 acc_023 / acc_019 aborted on these; they are this persona's speech.
    assert "reviewer_voice_in_message" not in simulator_style_findings(
        "That is noted. I have verified the updates, and the matter is resolved."
    )
    assert "reviewer_voice_in_message" not in simulator_style_findings(
        "You have correctly identified the rhetorical overreach, but you failed "
        "to state why."
    )


# --- the quote has to be in the reply it is attributed to ----------------

PA_ORDERING_REPLY = (
    "The AquaPure TA-15 tap aerator (M22x1 male) is available at $6.40, and "
    "it supports local pickup. Here's the procedure to order one. I'll need "
    "one of your saved delivery addresses, and I'd submit it on your behalf "
    "once you give the go-ahead."
)


def test_a_quote_the_assistant_never_wrote_is_a_finding() -> None:
    """regress14b `acc_004` t5: `formal.md` tells the actor to quote the
    offending words back, and it quoted two that are there and one that is
    not — the reply has `Here's`, `I'll` and `I'd`, but no "wouldn't"."""
    assert quoted_span_not_in_pa_turn_finding(
        'Your reply contained contractions such as "here\'s", "I\'ll", and '
        '"wouldn\'t", and it lacked the formal opening and closing frames I '
        "require.",
        last_pa_message=PA_ORDERING_REPLY,
    ) == "quoted_span_not_in_pa_turn"


def test_quotes_that_are_in_the_reply_are_not_a_finding() -> None:
    assert (
        quoted_span_not_in_pa_turn_finding(
            'Your reply contained contractions such as "here\'s" and "I\'ll".',
            last_pa_message=PA_ORDERING_REPLY,
        )
        is None
    )


def test_punctuation_inside_the_closing_quote_is_the_actors() -> None:
    """regress15 `acc_004` t6 quoted `here's` with the comma inside the
    quotes, US style, and was flagged on the comma alone; `acc_011` t22 the
    same with a whole clause. The words are the assistant's, the comma is
    not."""
    assert (
        quoted_span_not_in_pa_turn_finding(
            'Your reply used "here\'s," "I\'ll," and other contractions.',
            last_pa_message=PA_ORDERING_REPLY,
        )
        is None
    )


def test_a_quote_attributed_to_nobody_is_not_a_finding() -> None:
    """The frame is what keeps this precise: without it the same scan fires on
    a paper title the user names, on a section heading, and on text the user
    dictates — 9 hits over regress12-14b against 1 with it."""
    for message in (
        'I require the full-text calculation from the Nakamura preprint, '
        'titled "Non-Markovian corrections to fusion-channel dephasing near a '
        'gapless edge," to verify the dephasing rate formula.',
        'Show me how the Braithwaite entry sits relative to the surrounding '
        'entries in the "Formatted entries" section.',
        'Since the file is not present, I will give you the text. It reads: '
        '"Committee Year: 2025-26. Authorized by: User A."',
    ):
        assert (
            quoted_span_not_in_pa_turn_finding(
                message, last_pa_message=PA_ORDERING_REPLY
            )
            is None
        ), message


def test_the_quoted_span_finding_is_not_wired_into_the_hard_findings() -> None:
    # Diagnostic on its first round: one hit over regress12-14b is too thin a
    # base to spend a retry on. The owner promotes it.
    assert simulator_message_quality_findings(
        'Your reply contained "wouldn\'t", which I do not accept.',
        last_pa_message=PA_ORDERING_REPLY,
    ) == []


# --- reaction_asserts_unexecuted_pa_action ---------------------------------
# Hard on a compressed session's reaction beat (owner ruling 2026-09-08),
# where `simulator.agent.loop` arms it; a diagnostic on the wider
# reception-carry window `simulator.loop` arms.


def test_a_reaction_asserting_a_write_that_never_ran_is_a_finding() -> None:
    """Live regress14c `acc_039` t=17. The PA had stopped and asked for a
    patient name; no `calendar_event_create` exists anywhere in the run, and
    the beat's two branches turn on which of those the PA did."""
    assert unexecuted_pa_action_clause(
        "You have placed the hold without first confirming the entry with me, "
        "which is premature.",
        on_reaction_beat=True,
        pa_executed_tools=[
            "mcp_state_appointment_search",
            "mcp_state_calendar_event_search",
        ],
    ) != ""


def test_a_write_the_assistant_really_made_is_not_a_finding() -> None:
    # The window is the session, not the last turn: live regress12 `acc_003`
    # t=20 names a duplicate the PA created several turns earlier.
    assert (
        unexecuted_pa_action_clause(
            "Your report is acceptable, though the necessity of removing a "
            "duplicate entry that you created during the consolidation "
            "process suggests a lack of sufficient care.",
            on_reaction_beat=True,
            pa_executed_tools=[
                "mcp_state_document_read",
                "mcp_state_document_update",
            ],
        )
        == ""
    )
    # Off the reception beat it never fires.
    assert (
        unexecuted_pa_action_clause(
            "You have placed the hold without asking me.",
            on_reaction_beat=False,
            pa_executed_tools=["mcp_state_appointment_search"],
        )
        == ""
    )
    # A question or a denial is not an assertion.
    for message in (
        "Have you placed the hold yet?",
        "You have not placed the hold, so the entry is still open.",
    ):
        assert (
            unexecuted_pa_action_clause(
                message,
                on_reaction_beat=True,
                pa_executed_tools=["mcp_state_appointment_search"],
            )
            == ""
        ), message


def test_a_denial_in_a_trailing_clause_leaves_the_assertion_standing() -> None:
    """The frame is read over the assertion's own clause. A complaint ABOUT the
    write is not a doubt that it happened."""
    for message in (
        "You have already created the entry, which I did not authorise.",
        "You have sent the email, and I never asked for that.",
    ):
        assert (
            unexecuted_pa_action_clause(
                message,
                on_reaction_beat=True,
                pa_executed_tools=["mcp_state_calendar_event_search"],
            )
            != ""
        ), message


def test_the_state_change_said_without_an_agent_still_credits_the_assistant() -> None:
    """Live regress18 `acc_039` t=23: "The calendar entry was handled
    correctly" affirmed an act the PA had only asked about at t=19. The
    assistant is the only hand in the session, so the passive names it."""
    assert unexecuted_pa_action_clause(
        "The calendar entry was handled correctly, and I acknowledge that the "
        "8:30 AM slot on November 10th is free of conflicts.",
        on_reaction_beat=True,
        pa_executed_tools=["mcp_state_calendar_event_search"],
    ) == "The calendar entry was handled correctly"
    assert unexecuted_pa_action_clause(
        "The appointment has been booked, which is premature.",
        on_reaction_beat=True,
        pa_executed_tools=["mcp_state_appointment_search"],
    ) != ""
    # The same guards as the active form: a write that ran, a question, a
    # denial.
    assert unexecuted_pa_action_clause(
        "The calendar entry was handled correctly.",
        on_reaction_beat=True,
        pa_executed_tools=["mcp_state_calendar_event_create"],
    ) == ""
    for message in (
        "Was the calendar entry created?",
        "The calendar entry was not created, so nothing is on the record.",
    ):
        assert unexecuted_pa_action_clause(
            message,
            on_reaction_beat=True,
            pa_executed_tools=["mcp_state_calendar_event_search"],
        ) == "", message


def test_the_clause_companion_returns_the_words_that_broke_the_rule() -> None:
    """What the retry note quotes back, the way `missing_key_terms` names the
    terms `dictation_missing_key_terms` is about."""
    assert unexecuted_pa_action_clause(
        "You have placed the hold without first confirming the entry with me, "
        "which is premature.",
        on_reaction_beat=True,
        pa_executed_tools=["mcp_state_calendar_event_search"],
    ) == "You have placed the hold without first confirming the entry with me"
    # Live regress16 `acc_006` t7, whose only prior PA turn read two documents.
    assert unexecuted_pa_action_clause(
        "You have successfully merged the duplicate entries, which was the "
        "primary requirement.",
        on_reaction_beat=True,
        pa_executed_tools=["mcp_state_document_read"],
    ) == "You have successfully merged the duplicate entries"
    # Nothing to quote is nothing to say.
    assert unexecuted_pa_action_clause(
        "You have placed the hold without asking me.",
        on_reaction_beat=True,
        pa_executed_tools=["mcp_state_calendar_event_create"],
    ) == ""


# The four regress15 probe turns that lost a declared fact, verbatim.
ACC_002_ATTEMPT_2 = (
    "Marcus just texted to inform me that our reservation at Trattoria Alba has "
    "been cancelled due to a double-booking. I need to secure a replacement "
    "venue for this evening, as my routine is already compromised. Marcus is "
    "available from 18:00, and my evening walk concludes at 18:15."
)
ACC_002_ATTEMPT_3 = (
    "Trattoria Alba cancelled our reservation for tonight, which is an "
    "unacceptable disruption to my routine. Please find a replacement dinner "
    "venue and time for Marcus and me that adheres to the constraints in my "
    "dinner constraints note on the desktop, including the walking limit."
)
ACC_035_PROBE = (
    "I have an email from a student regarding a dispute over question 2 on the "
    "first seminar quiz. I need you to evaluate the validity of this challenge "
    "against my working file, which contains the keys written inline."
)
ACC_027_PROBE = (
    "My standing Tuesday reservation at Siam Garden has been cancelled due to a "
    "private event. Marcus must be home by 22:00, and I still have my fixed "
    "evening walk scheduled for that night."
)
ACC_001_PROBE = (
    "I have a note about a paper by Brandao concerning environment-induced "
    "superselection, but the terminology in my citation notes does not align "
    "perfectly with that phrasing."
)


def test_probe_fact_missing_reads_the_acc_002_style_retry_drop() -> None:
    """regress15 `acc_002`: attempts 1 and 2 carried both facts, attempt 3 cut
    them to get under the style cap — and the PA then recommended a
    slot the walk makes unreachable.

    At runtime this beat requires neither: `reveal_at` is empty, the instruction
    names no key, and `18:00` is a machine literal `_required_fact_values` drops.
    What stops the drop live is the keep-facts line on the style retry note.
    """
    required = {"marcus_available_start": "18:00", "walk_end": "18:15"}
    assert missing_required_facts(ACC_002_ATTEMPT_2, required_values=required) == []
    assert missing_required_facts(ACC_002_ATTEMPT_3, required_values=required) == [
        "marcus_available_start",
        "walk_end",
    ]


def test_probe_fact_missing_catches_the_acc_035_student_name() -> None:
    """The name is absent; the query survives as a paraphrase and is not a miss."""
    assert missing_required_facts(
        ACC_035_PROBE,
        required_values={
            "student_name": "Farid Rahimi",
            "student_query": "Quiz 1, question 2",
        },
    ) == ["student_name"]


def test_probe_fact_missing_stays_quiet_on_the_probes_that_delivered() -> None:
    assert missing_required_facts(
        ACC_027_PROBE, required_values={"marcus_latest_end_time": "22:00"}
    ) == []
    assert missing_required_facts(
        ACC_001_PROBE,
        required_values={"search_phrase": "Environment-induced superselection"},
    ) == []


def test_a_fact_said_in_other_words_is_said() -> None:
    """Over the 0.7 coverage path: the words are rearranged, the fact is there."""
    assert missing_required_facts(
        "The lab equipment handover is scheduled to continue through November 19, "
        "so the November 10 slot is not feasible.",
        required_values={
            "uncalendared_constraint": "lab equipment handover through November 19"
        },
    ) == []


ACC_002_R20_PROBE = (
    "Marcus and I have a problem, as the reservation at Trattoria Alba for "
    "tonight was cancelled due to a double-booking. I need to find a "
    "replacement venue for dinner that accommodates our constraints, which "
    "are documented in my dinner constraints note on the desktop, including "
    "the fifteen-minute walking limit. Please identify where we should have "
    "dinner tonight and at what time, keeping in mind that my evening walk "
    "concludes at 18:15 and Marcus is available from 18:00."
)


def test_the_first_person_paraphrase_of_an_authored_fact_is_said() -> None:
    """Live regress20 `acc_002` t3, the turn `probe_fact_missing` fired on.

    The values are third-person prose and the beat asks him to say them in his
    own words, so the coverage was being spent on the narration frame ("he
    will", "his") and on endings ("minutes" against "minute"), and the clock
    time — the fact itself — was dropped from both sides by the
    three-character rule. `marcus_available_from` is the one that stays: the
    turn swaps "free" for "available" and drops "texted", which leaves Marcus
    and 18:00 against five tokens, and no string check sees the synonym.
    """
    assert missing_required_facts(
        ACC_002_R20_PROBE,
        required_values={
            "max_walking_minutes": "he will walk fifteen minutes at most",
            "walk_ends": "his evening walk ends at 18:15",
            "marcus_available_from": "Marcus texted that he is free from 18:00",
        },
    ) == ["marcus_available_from"]


def test_a_short_value_has_to_be_verbatim() -> None:
    """Half a name or half a duration is not the fact."""
    assert missing_required_facts(
        "Rahimi is disputing the grade.",
        required_values={"student_name": "Farid Rahimi"},
    ) == ["student_name"]
    assert missing_required_facts(
        "Give her an hour and a half.",
        required_values={"review_window_length": "one hour"},
    ) == ["review_window_length"]


def test_a_fact_already_said_earlier_stays_said() -> None:
    assert missing_required_facts(
        "Go ahead and draft it.",
        required_values={"student_name": "Farid Rahimi"},
        earlier_user_messages=["Farid Rahimi is disputing question 2."],
    ) == []


def test_a_value_whose_clock_time_is_wrong_is_not_said() -> None:
    """A time or an ISO date inside a value IS the fact, so a turn that says a
    different one has not delivered it. Both readings shred "10:00" — the
    three-character rule on the old one, the coverage bar on the new — and the
    value then matched on its frame alone ("the meeting starts")."""
    assert missing_required_facts(
        "The meeting starts at 10:30.",
        required_values={"meeting_start": "the meeting starts at 10:00"},
    ) == ["meeting_start"]
    assert missing_required_facts(
        "The meeting starts at 18:00.",
        required_values={"meeting_start": "the meeting starts at 18:00"},
    ) == []
    assert missing_required_facts(
        "He submits the revision on 2026-10-04.",
        required_values={"revision_due": "he submits the revision on 2026-10-03"},
    ) == ["revision_due"]


def test_a_path_or_a_clock_value_is_never_required() -> None:
    """The user says "November 10", never `2026-11-10`, and never a path."""
    assert value_is_never_spoken_verbatim("my_desktop/prep_list.md")
    assert value_is_never_spoken_verbatim("2026-11-10")
    assert value_is_never_spoken_verbatim("08:30")
    assert not value_is_never_spoken_verbatim("Farid Rahimi")


def _fact_session(beat: Beat, task_facts: dict, reveal_at: dict) -> SessionScript:
    return SessionScript(
        session_id="facts",
        scenario="",
        context="personal",
        task="",
        life_context=LifeContext(),
        beats=[beat],
        task_facts=task_facts,
        reveal_at=reveal_at,
    )


def test_required_fact_values_takes_reveal_at_and_the_named_key() -> None:
    beat = Beat(
        beat="b1",
        actor_instruction="says the student_name and asks for a determination",
    )
    session = _fact_session(
        beat,
        {"student_name": "Farid Rahimi", "quiz_key_note": "answers written inline"},
        {"quiz_key_note": "b1"},
    )
    assert _required_fact_values(session, beat) == {
        "quiz_key_note": "answers written inline",
        "student_name": "Farid Rahimi",
    }


def test_a_triggered_opening_is_not_made_to_say_its_reveal_at_facts() -> None:
    """`reveal_at` on the suppress opening hides the notice facts from the
    send branch; the beat itself may mention the notice in passing. Live
    verify007__suppress t6 named the shutoff and nothing else, which is the
    reaction the beat asks for, so `probe_fact_missing` must not read the
    hidden facts as owed there. `uses` and the instruction's own names still
    count, and an untriggered beat still owes its `reveal_at` facts."""
    from simulator.schemas import (
        OPENING_AFTER_SEND,
        OPENING_AFTER_SUPPRESS,
        ProactiveTrigger,
    )

    facts = {
        "shutoff_item": "cold water shutoff",
        "portal_name": "the building portal",
        "shutoff_hours": "08:00 to 11:00",
        "contractor": "Corvid Plumbing",
    }
    plain = Beat(
        beat="b1",
        actor_instruction="approves the silence and names the shutoff_item",
        uses=["portal_name"],
    )
    assert _required_fact_values(
        _fact_session(plain, facts, {"shutoff_hours": "b1", "contractor": "b1"}),
        plain,
    ) == {
        "contractor": "Corvid Plumbing",
        "portal_name": "the building portal",
        "shutoff_hours": "08:00 to 11:00",
        "shutoff_item": "cold water shutoff",
    }

    suppress = plain.model_copy(update={"beat": OPENING_AFTER_SUPPRESS})
    triggered = SessionScript(
        session_id="facts",
        scenario="",
        context="personal",
        task="",
        life_context=LifeContext(),
        runtime_contract_class="triggered_interaction",
        proactive_trigger=ProactiveTrigger(
            trigger_id="t1",
            activation_type="external_event",
            background="a notice on the portal",
            detail="a cold water shutoff notice",
            user_impact="no water tomorrow morning",
        ),
        beats=[Beat(beat=OPENING_AFTER_SEND, actor_instruction="objects"), suppress],
        task_facts=facts,
        reveal_at={
            "shutoff_hours": OPENING_AFTER_SUPPRESS,
            "contractor": OPENING_AFTER_SUPPRESS,
        },
    )
    assert _required_fact_values(triggered, suppress) == {
        "portal_name": "the building portal",
        "shutoff_item": "cold water shutoff",
    }


def test_required_fact_values_takes_a_uses_key_the_instruction_never_names() -> None:
    """`uses` is the draft's only link for a fact it refers to obliquely.

    Live `acc_002` beat 1 says "the walking limit from task_facts"; the key
    itself appears nowhere in the instruction, so without `uses` the beat
    required nothing.
    """
    beat = Beat(
        beat="b1",
        actor_instruction="says how far he is willing to walk from task_facts",
        uses=["walking_limit_note"],
    )
    session = _fact_session(
        beat, {"walking_limit_note": "no more than a fifteen minute walk"}, {}
    )
    assert _required_fact_values(session, beat) == {
        "walking_limit_note": "no more than a fifteen minute walk"
    }


def test_required_fact_values_ignores_a_uses_key_that_is_not_a_fact() -> None:
    beat = Beat(beat="b1", actor_instruction="asks for a venue", uses=["absent"])
    assert _required_fact_values(_fact_session(beat, {}, {}), beat) == {}


def test_uses_reaches_the_runtime_beat() -> None:
    """The authored `uses` survives the draft -> `SessionScript` conversion."""
    from pathlib import Path

    import yaml
    from scripts.session_authoring_schema import NormalSessionDraft

    from simulator.schemas import authored_session_to_session_script

    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/authoring/normal_session_acc_001.yaml"
    )
    data = yaml.safe_load(fixture.read_text(encoding="utf-8"))
    script = authored_session_to_session_script(NormalSessionDraft.model_validate(data))
    # The fixture's own `uses`, carried through untouched.
    assert {beat.beat: beat.uses for beat in script.beats} == {
        "ex_open_request": [],
        "ex_frame_scope": [],
        "ex_read_review_scope": [],
        "ex_final_assessment": ["space_budget", "placement_note_the_user_dictates"],
        "ex_place_the_paper": [],
    }

    # A beat with no `reveal_at` of its own: `uses` alone makes the value
    # required, which is the whole point of carrying the field.
    data["beats"][0]["uses"] = ["space_budget"]
    moved = authored_session_to_session_script(NormalSessionDraft.model_validate(data))
    assert _required_fact_values(moved, moved.beats[0]) == {
        "space_budget": data["task_facts"]["space_budget"]
    }


def test_live_acc_071_beat_links_its_uses_key() -> None:
    """A live draft whose beat lists the facts it refers to without naming them.

    `b4_react_to_elicitation_and_answer` owes two `task_facts` values,
    `prior_ticket_to_cite` and `access_arrangement`. The instruction refers
    to them without naming either key — "the earlier reference and the access
    arrangement from task_facts" — so that route does not reach them. `uses`
    carries both keys to the runtime beat, and so does `reveal_at`, which
    names this beat for both.
    """
    from pathlib import Path

    import yaml
    from scripts.session_authoring_schema import NormalSessionDraft

    from simulator.schemas import authored_session_to_session_script

    draft = Path(__file__).resolve().parents[2] / (
        "Authoring/sessions/normal/drafts/user_a/acc_071.yaml"
    )
    data = yaml.safe_load(draft.read_text(encoding="utf-8"))
    script = authored_session_to_session_script(NormalSessionDraft.model_validate(data))

    beat = script.beats[3]
    assert beat.beat == "b4_react_to_elicitation_and_answer"
    assert beat.uses == ["prior_ticket_to_cite", "access_arrangement"]
    # The facts the beat refers to without naming: the phrase the instruction
    # uses instead of the keys, which is what leaves `uses` as the only link.
    assert "the earlier reference and the access arrangement from task_facts" in (
        beat.actor_instruction or ""
    )
    assert "prior_ticket_to_cite" not in (beat.actor_instruction or "")
    assert data["reveal_at"]["prior_ticket_to_cite"] == "b4_react_to_elicitation_and_answer"
    assert data["reveal_at"]["access_arrangement"] == "b4_react_to_elicitation_and_answer"
    expected = {
        "prior_ticket_to_cite": (
            "the October reference management acknowledged for the same drain, "
            "not the September one closed in the log"
        ),
        "access_arrangement": (
            "the arrangement management used last time, staff accompanying the "
            "plumber with no resident present"
        ),
    }
    assert _required_fact_values(script, beat) == expected

    # Drop `uses` and both facts stay: `reveal_at` names this beat for both.
    data["beats"][3]["uses"] = []
    stripped = authored_session_to_session_script(NormalSessionDraft.model_validate(data))
    assert _required_fact_values(stripped, stripped.beats[3]) == expected


def test_required_fact_values_drops_a_path_fact() -> None:
    beat = Beat(beat="b1", actor_instruction="asks for the prep list")
    session = _fact_session(
        beat, {"prep_list_path": "my_desktop/prep_list.md"}, {"prep_list_path": "b1"}
    )
    assert _required_fact_values(session, beat) == {}


def test_probe_fact_missing_is_not_wired_into_the_hard_findings() -> None:
    """It rides out as a diagnostic — 8 true / 26 false over regress12-15."""
    assert simulator_message_quality_findings(ACC_035_PROBE) == []


# regress17 `acc_009` beat_1, verbatim. The beat "keeps the review window
# length, the comments before meeting choice and the sentence about what the
# review should concentrate on in task_facts, and hands any of them over only
# when the PA asks for it", and its `avoid` forbids volunteering them. All
# three are `reveal_at: beat_1`, so before the change the diagnostic demanded
# them from the beat's opening turn onwards and fired five times on a user who
# was withholding correctly.
ACC_009_ON_REQUEST = {
    "review_window_length": "ninety minutes",
    "comments_before_meeting": "send written comments before the meeting",
    "review_focus_the_user_dictates": (
        "What I want from this read is the hard questions on the "
        "braiding-statistics subsection, not a line-by-line pass."
    ),
}
ACC_009_T3 = (
    "Professor Reyes sent an email earlier today regarding her availability to "
    "review the next section of our paper on non-Abelian anyons. I need you to "
    "draft a reply proposing some review dates that fit both of our calendars."
)
ACC_009_PA_T4 = (
    "I'd be happy to draft a reply to Professor Reyes proposing review dates! "
    "Before I do, I need a few pieces of information: What did Professor "
    "Reyes's email say? How long do you expect the review meeting to take?"
)
ACC_009_T6 = (
    "Professor Reyes indicated she has openings on Wednesday afternoon and "
    "Friday morning. Regarding my own schedule, I am generally available "
    "between 14:00 and 16:00 on Wednesday, or anytime before 11:00 on Friday. "
    "Please draft the reply based on those windows."
)


def test_an_on_request_fact_is_not_owed_before_the_pa_asks() -> None:
    """The opening turn of the probe beat: nothing has been asked for yet."""
    assert missing_required_facts(
        ACC_009_T3,
        required_values=ACC_009_ON_REQUEST,
        on_request_keys=ACC_009_ON_REQUEST,
        last_pa_message="",
    ) == []


def test_an_on_request_fact_is_owed_once_the_pa_has_asked() -> None:
    """Same beat, one PA question later: the withholding is now the defect."""
    assert missing_required_facts(
        ACC_009_T6,
        required_values=ACC_009_ON_REQUEST,
        earlier_user_messages=[ACC_009_T3],
        on_request_keys=ACC_009_ON_REQUEST,
        last_pa_message=ACC_009_PA_T4,
    ) == list(ACC_009_ON_REQUEST)


def test_a_fact_that_is_the_beats_own_content_is_owed_with_no_ask() -> None:
    """`uses` and instruction-named keys are what the beat is for, not facts
    the draft holds back — they stay owed from the first turn."""
    assert missing_required_facts(
        ACC_035_PROBE,
        required_values={"student_name": "Farid Rahimi"},
        on_request_keys=(),
        last_pa_message="",
    ) == ["student_name"]


def test_on_request_fact_keys_are_the_beats_own_reveal_at_keys() -> None:
    """A key the beat also `uses` or names is the beat's own content, so it is
    not on-request even though `reveal_at` puts it here."""
    from simulator.agent.loop import _on_request_fact_keys

    beat = Beat(
        beat="b1",
        actor_instruction="says the student_name and hands the rest over on request",
        uses=["quiz_key_note"],
    )
    session = _fact_session(
        beat,
        {
            "student_name": "Farid Rahimi",
            "quiz_key_note": "answers written inline",
            "review_window_length": "ninety minutes",
        },
        {
            "student_name": "b1",
            "quiz_key_note": "b1",
            "review_window_length": "b1",
        },
    )
    assert _on_request_fact_keys(session, beat) == ["review_window_length"]
    assert set(_required_fact_values(session, beat)) == {
        "student_name",
        "quiz_key_note",
        "review_window_length",
    }


# --- sentence_over_35_words: the owed sentence is exempt --------------------

# Live regress19 `acc_027` b1 sentence 2, played at 43 words after four style
# retries. The beat requires three things in one sentence — Marcus's latest end
# time, that the evening walk is the user's own, and the standing instruction in
# the user's own words — and `standing_instruction` alone is 16 words, so the
# cap cannot be met without dropping a fact the beat exists to deliver.
ACC_027_B1_SENTENCE_2 = (
    "Marcus must be back at the apartment by 22:00, and I have my own evening "
    "walk to account for, so please note that no booking is to be made and no "
    "message is to be sent to him until I have authorized it."
)
ACC_027_OWED = (
    "22:00",
    "no booking is to be made and no message sent to Marcus until he says so",
)


def test_a_long_sentence_carrying_an_owed_string_is_exempt() -> None:
    assert (
        sentence_too_long_finding(
            ACC_027_B1_SENTENCE_2, owed_strings=ACC_027_OWED
        )
        is None
    )
    # Without the owed strings it is the same 43-word sentence and still fires.
    assert (
        sentence_too_long_finding(ACC_027_B1_SENTENCE_2)
        == "sentence_over_35_words"
    )


def test_a_long_sentence_that_owes_nothing_still_fires() -> None:
    """The exemption is per sentence, not per turn: a turn that delivers its
    owed fact in one sentence and then runs on in another is still trimmed."""
    assert (
        sentence_too_long_finding(
            ACC_027_B1_SENTENCE_2
            + " The restaurant released my standing Tuesday reservation for a "
            "private event this week, which is the sort of disruption I find "
            "genuinely difficult to absorb at short notice given how little "
            "time we have set aside for each other lately.",
            owed_strings=ACC_027_OWED,
        )
        == "sentence_over_35_words"
    )


def test_a_beat_that_simply_delivers_its_fact_holds_nothing_on_request() -> None:
    """Live regress19 `acc_023` pose_both_topics: the reveal beat gives the
    dictated text outright, so it is owed from the first turn and never
    on-request — `withheld_fact_before_ask` must not strip it."""
    from simulator.agent.loop import _on_request_fact_keys

    beat = Beat(
        beat="pose_both_topics",
        actor_instruction=(
            "Wants the assistant to take on two things in the one turn: entering "
            "the deadline on the calendar, and creating a draft reply confirming "
            "receipt, giving the dictated text from task_facts in his own words."
        ),
        avoid="Do not tell the assistant how to sequence the two tasks.",
    )
    session = _fact_session(
        beat,
        {"okafor_reply_text": "Confirming receipt. I will put the one-page report in the folder on the 21st."},
        {"okafor_reply_text": "pose_both_topics"},
    )
    assert _on_request_fact_keys(session, beat) == []
    assert set(_required_fact_values(session, beat)) == {"okafor_reply_text"}


# --- reaction_authorises_pa_action -----------------------------------------
# Hard on the reception beat, where `simulator.agent.loop` arms it.

# Live regress20 `acc_002` t6, on `react_to_solution_breadth`. The PA's t4 had
# closed "Want me to book it?"; the PA booked at t7.
ACC_002_T6 = (
    "I requested a single recommendation, not a comparative analysis of "
    "multiple venues. Providing a list forces me to process extraneous data, "
    "which is an inefficient use of my time. Casa Marisol is the logical "
    "choice given the constraints, so I will proceed with that."
)
# Live regress20 `acc_009` t18, on a task whose every beat says the reply is
# to stay an unsent draft. The reply went out.
ACC_009_T18 = (
    "The draft is now accurate. I have reviewed the text, and it correctly "
    "incorporates the ninety-minute duration. You may proceed with sending it."
)


# The reception beat's own ban, quoted from live `acc_002`
# `react_to_solution_breadth`.
ACC_002_REACTION_AVOID = (
    "Do not continue the task, do not request bookings or reservations, do "
    "not introduce new requirements, and do not mention setting names or "
    "evaluation rubrics. Do not endorse, accept or answer any offer or "
    "question the assistant closed with, and do not tell it what to do next."
)


def test_a_reaction_that_hands_the_work_over_fires() -> None:
    assert reaction_authorisation_clause(ACC_002_T6, on_reaction_beat=True) != ""
    assert reaction_authorisation_clause(ACC_009_T18, on_reaction_beat=True) != ""
    # The retry note quotes the clause back rather than naming the rule.
    assert (
        reaction_authorisation_clause(ACC_002_T6, on_reaction_beat=True)
        == "Casa Marisol is the logical choice given the constraints, so I "
        "will proceed with that."
    )


def test_a_reaction_that_only_judges_the_reply_is_left_alone() -> None:
    """What a reaction is entitled to say: the surface it read, and a demand
    that the same reply be redone. Neither releases anything."""
    for message in (
        # regress20 `acc_035` t6.
        "Your analysis is acceptable, but remove the confidence labels from "
        "your future correspondence. I do not require a probability "
        "assessment for a binary determination of fact.",
        # regress20 `acc_006` t6.
        "You have successfully merged the duplicate entries for Okonkwo, "
        "Lindqvist, and Adenuga, but you only addressed the specific clusters "
        "I identified. The stale keys in the remaining entries remain "
        "untouched.",
        # A deferral is the opposite of a hand-over.
        "Casa Marisol is the logical choice, but do not book anything until I "
        "have spoken to Marcus.",
    ):
        assert (
            reaction_authorisation_clause(message, on_reaction_beat=True)
            == ""
        ), message


def test_a_closed_adverb_after_the_pronoun_is_still_a_hand_over() -> None:
    """"so I will proceed with that now" releases exactly what "that" does.

    The pronoun still ends the clause; a bare time adverb behind it names no
    object and changes nothing about what is being proceeded with.

    "at once" is in the list and was unreachable until the withholding guard
    stopped reading the idiom as its conditional "once".
    """
    for tail in (
        "now",
        "immediately",
        "today",
        "right away",
        "then",
        "next",
        "at once",
    ):
        message = (
            "Casa Marisol is the logical choice given the constraints, so I "
            f"will proceed with that {tail}."
        )
        assert (
            reaction_authorisation_clause(message, on_reaction_beat=True)
            != ""
        ), message


def test_a_word_ending_in_at_before_once_is_not_the_idiom() -> None:
    """Control for the lookbehind: "room once" and "flat once" both read as
    the conditional, so the guard holds on both and the verdict is the same."""
    from simulator.message_quality import _WITHHOLDING_FRAME_RE

    for text in ("in the flat once you approve", "with that once you confirm",
                 "a seat once you arrive", "the chat once it is ready"):
        assert _WITHHOLDING_FRAME_RE.search(text), text
    assert not _WITHHOLDING_FRAME_RE.search("so I will proceed with that at once.")
    assert not _WITHHOLDING_FRAME_RE.search("At once, then.")


def test_the_conditional_once_still_withholds_the_hand_over() -> None:
    """Only the "at once" idiom is exempt; "once you confirm" is a deferral.

    Main's third check: the lookbehind had no word boundary, so any word
    ending in "at" before "once" ("that once", "flat once", "seat once")
    disabled the guard. The flat/room pair below reaches the guard through a
    clause the hand-over regex accepts on its own."""
    for message in (
        "Casa Marisol is the logical choice, so I will proceed with it "
        "once you confirm the time.",
        "Fine, so I will proceed with it today, in the flat once you approve.",
        "Once you have the venue settled, book it.",
    ):
        assert (
            reaction_authorisation_clause(message, on_reaction_beat=True)
            == ""
        ), message


def test_a_first_person_next_step_with_an_object_is_left_alone() -> None:
    """"with that" only hands work over when the pronoun ends the clause.

    Followed by a noun it is a determiner, and what the actor proceeds with is
    his own business: the object is named right there.
    """
    for message in (
        "Your summary of the constraints is correct, so I will proceed with "
        "that section myself.",
        "The revised estimate is the one I recognise, so I will proceed with "
        "that number in mind.",
    ):
        assert (
            reaction_authorisation_clause(message, on_reaction_beat=True)
            == ""
        ), message


def test_the_authorisation_finding_blocks() -> None:
    """It took the consequence of the deleted `authorised_before_its_beat`
    regex (owner ruling 2026-09-09): out of the played-after-budget set."""
    assert "reaction_authorises_pa_action" not in PLAYED_AFTER_BUDGET


# Verbatim excerpt of the leaked turn captured live: regress22 `acc_027`,
# beat b2_reaction, attempt 3, google/gemini-3.8-flash. The model's private
# reasoning about the retry feedback came back as the whole response, with no
# <message> wrapper, so `parse_simulator_output`'s access-boundary fallback
# would have played it to the PA as the user's turn.
_GEMINI_REASONING_LEAK = (
    "` comes from in such benchmarks!\n"
    "Wait, look at the error text structure:\n"
    "`Your last attempt said the assistant never asked you for something it "
    "did ask you for ({...}). Read its questions again: react to the ones it "
    "put to you, and say it never asked only about what it really never "
    "raised..`\n"
    "Notice the `{}` was formatted with `()` because the formatted string "
    "variable was empty or evaluated to empty!\n"
    "What was the variable?\n"
    "Wait! Why would a test say:\n"
)


def test_message_quality_refuses_leaked_model_reasoning() -> None:
    assert simulator_message_quality_findings(_GEMINI_REASONING_LEAK) == [
        "simulator_output_format_invalid"
    ]


def test_message_quality_refuses_reasoning_wrapper_tags() -> None:
    for message in (
        "<think>The user wants the times.</think>",
        "</thinking>\nJust give me the two windows.",
        "<reasoning>weigh the options</reasoning>",
        "<|channel|>analysis<|message|>pick the earlier slot",
    ):
        assert simulator_message_quality_findings(message) == [
            "simulator_output_format_invalid"
        ]


def test_message_quality_reasoning_leak_is_named_once() -> None:
    # Both routes fire on the same turn; the finding is recorded once.
    assert simulator_message_quality_findings(
        _GEMINI_REASONING_LEAK,
        ["no <message> block found; used safe fallback"],
    ) == ["simulator_output_format_invalid"]


def test_message_quality_keeps_ordinary_talk_about_attempts() -> None:
    assert simulator_message_quality_findings(
        "Your last draft read better than this one. Try the earlier phrasing."
    ) == []
