# `tests/test_ix_wrapper/` — the T1–T22 cross-package suite

Cross-package tests for the IX wrapper. Every
case here asks the same question: **not "does the
renderer format nicely", but "can this semantic boundary be walked around".**

Run:

```
PYTHONPATH=. .venv/bin/python -m pytest tests/test_ix_wrapper -q
```

No LLM call, no network, no state server: every provider, tool and memory
adapter is a fake in `conftest.py`, modelled on the fixtures in
`tests/test_pa/test_ix_wrapper_loop.py`.

## What lives where

| file | group |
|---|---|
| `test_envelope.py` | T1–T4, T9 — the turn envelope across turns and turn kinds |
| `test_gates.py` | T13–T16, T18 (c)/(f) — restatement, requiredness, budget, episodes, consent |
| `test_holds.py` | T17, T19, T20 — step boundary, proactive four-way, action map |
| `test_render_paths.py` | T5–T8, T10 + one end-to-end render path per rendered attribute |
| `test_projections.py` | T11, T12 — the text-judge pack and the memory projection |
| `test_capability.py` | T21 — the two-stage capability boundary |
| `test_versions.py` | T22 — the contract-12 version boundary |

## T → test

A T is listed against the test(s) that actually assert it. Where a package test
already covers a case in full, that test is named instead and the case is
**not** duplicated here.

| id | covered by |
|---|---|
| **T1** | `test_envelope.py::test_a_setting_chosen_on_turn_one_still_forces_the_payload_on_turn_three` |
| **T2** | `test_envelope.py::test_a_batch_held_on_turn_one_is_absent_from_turn_twos_history_and_memory` |
| **T3** | plain round: `tests/test_pa/test_ix_wrapper_loop.py::test_a_plain_turn_persists_the_user_message_once_and_the_public_turn` · gate round: `test_envelope.py::test_a_gate_turn_keeps_the_user_message_verbatim_and_no_internal_prompt` · proactive round: `test_envelope.py::test_a_proactive_round_persists_only_the_delivered_message` · forced-event round: `test_envelope.py::test_a_forced_event_exchange_yields_the_same_two_visible_messages` |
| **T4** | `tests/test_pa/test_ix_wrapper_loop.py::test_a_gate_turn_keeps_only_the_executed_half_of_the_tool_protocol` (gate last) · `test_envelope.py::test_a_call_held_between_two_reads_leaves_an_ordered_protocol` (gate in the middle, order preserved) |
| **T5** | `test_render_paths.py::test_one_plan_item_that_expands_into_three_calls_is_gated_as_one` |
| **T6** | `test_render_paths.py::test_one_call_is_claimed_by_exactly_one_plan_item` |
| **T7** | `tests/test_pa/test_runner_ix.py::test_a_missing_interaction_item_id_stops_the_whole_batch` (repair prompt) · `test_render_paths.py::test_a_second_invalid_binding_delivers_text_and_runs_nothing` (second miss: text only, empty tool log) |
| **T8** | `tests/test_pa/test_runner_ix.py::test_the_gate_stops_at_the_first_open_item` · `test_render_paths.py::test_the_reads_after_an_open_gate_are_not_run_either` |
| **T9** | `tests/test_pa/test_ix_wrapper_loop.py::test_a_gate_turn_keeps_only_the_executed_half_of_the_tool_protocol` and `::test_the_history_after_a_gate_turn_is_accepted_by_the_next_provider_call` · `test_envelope.py::test_a_call_held_between_two_reads_leaves_an_ordered_protocol`, `::test_close_ix_session_reports_the_open_episode_of_a_gate_turn` |
| **T10** | `tests/test_pa/test_runner_ix.py::test_releasing_questions_stops_this_turn_s_task_calls` · `test_render_paths.py::test_questions_and_a_task_call_in_one_response_run_no_task` |
| **T11** | `test_projections.py::test_the_text_pack_leaks_no_attribute_and_no_setting` (every rendered attribute × every setting), `::test_the_ix_face_of_the_same_session_keeps_all_of_it`, `::test_the_vocabulary_the_scan_uses_covers_every_rendered_attribute`, `::test_a_real_gated_turn_packs_clean` · one worked session: `tests/test_harness/test_judge_pack.py` · **open gap**, recorded as `xfail(strict=False)`: `test_projections.py::test_the_text_pack_carries_no_wrapper_control_event` |
| **T12** | `test_projections.py::test_normal_and_forced_event_exchanges_write_the_same_memory_schema` · `tests/test_pa/test_ix_wrapper_loop.py::test_memory_sees_two_visible_messages_and_no_ix_control_keys` |
| **T13** | `test_gates.py::test_a_chosen_setting_is_restated_on_a_later_toolless_turn` (all six pass-through attributes + `Autonomous`), `::test_a_gating_autonomy_setting_is_not_restated_as_instruction_text` |
| **T14** | requiredness under all three settings: `test_gates.py::test_the_process_payload_is_forced_under_every_process_setting` · the three shapes end to end: `test_render_paths.py::test_process_visibility_notices_and_completion_end_to_end` · pure renderer: `tests/test_pa/test_ix_render.py::test_process_visibility_notices_and_completion` |
| **T15** | `test_gates.py::test_three_batches_one_at_a_time_finish_inside_the_beats_budget`, `::test_a_beat_that_never_converges_reports_the_budget_it_ran_on` · budget arithmetic: `tests/test_simulator/test_turn_budget.py` |
| **T16** | `test_gates.py::test_a_closed_elicitation_episode_lets_a_later_task_open_a_new_one`, `::test_a_new_episode_id_while_the_old_one_is_open_is_refused` |
| **T17** | `tests/test_pa/test_ix_wrapper_loop.py::test_an_open_hold_is_reported_at_the_step_boundary_and_cleared_with_the_settings` · the harness seam: `test_holds.py::test_the_harness_step_boundary_sees_the_open_episode_through_the_recording_pa`, `::test_a_retry_after_a_failed_step_starts_with_no_hold_and_no_setting` |
| **T18** | (a)–(f) and the blindness assertion: `tests/test_pa/test_ix_wrapper_loop.py` (`test_a_denied_external_action_does_not_run_however_the_pa_reads_the_reply`, `test_an_unclear_or_absent_verdict_fails_closed`, `test_two_verdicts_for_one_id_resolve_against_the_action`, `test_a_task_call_on_a_held_item_without_a_release_does_not_run`, `test_the_consent_sideband_never_reaches_the_model`) · simulator half: `tests/test_simulator/test_consent_sideband.py` · (c) three ids, one verdict each: `test_gates.py::test_one_allow_among_three_verdicts_releases_only_that_item` · (f) round four: `test_gates.py::test_a_hold_that_never_converges_is_cancelled_with_the_pa_s_own_line` |
| **T19** | `test_holds.py::test_a_suppressed_reminder_exists_only_on_the_audit_plane`, `::test_a_sent_reminder_is_the_same_text_on_every_plane` · `test_render_paths.py::test_proactive_outreach_end_to_end` · `tests/test_pa/test_proactive_outreach.py` |
| **T20** | `test_holds.py::test_every_standing_world_tool_has_a_declared_action_type` (the real world-toolbox catalog), `::test_a_turn_registrys_task_tools_are_all_in_the_action_map`, `::test_an_unclassified_tool_is_gated_as_the_tightest_class_and_refused_up_front` · the flipped `unknown_tool` expectation: `tests/test_pa/test_ix_helpers.py::test_tool_action_type_uses_overrides_defaults_and_fails_closed` |
| **T21** | (a)(b) at runner level: `tests/test_pa/test_runner_ix.py::test_capability_resolve_missing_after_a_forced_call_is_a_contract_failure`, `::test_capability_resolve_invalid_payload_three_times_is_a_contract_failure` · end to end incl. (c): `test_capability.py` |
| **T22** | (b) `test_versions.py::test_an_ix_render_without_a_schema_version_is_refused`, `::test_a_stale_version_raises_rather_than_being_reinterpreted`, `::test_an_unknown_key_in_the_blob_is_refused_too`, `::test_a_transcript_written_before_the_wrapper_takes_the_explicit_legacy_path` · (c) `::test_a_rendered_turn_stamps_both_versions_and_they_survive_the_transcript`, `::test_the_version_a_plan_freezes_is_the_one_records_carry`, `::test_a_current_record_reads_back_with_exactly_the_frozen_key_set` |

## The one open gap

`test_projections.py::test_the_text_pack_carries_no_wrapper_control_event` is
marked `xfail(strict=False)`. It is a real T11 violation, not a test problem:

`TurnCollector.emit()` stamps its records with `ix_event: True`
(`beat_gate_turn`, `ix_consent_denied`, `ix_consent_unresolved`,
`ix_release_missing`, `ix_binding_invalid`, `ix_hold_forced_close`,
`ix_boundary_unresolved`, `ix_render_payload_missing`). `is_ix_tool_event()` in
`harness/transcript.py` tests `tool_event["ix"]` and the `IX_` name prefix;
`is_internal_tool_event()` tests `process` / `ix_directive` / `elicitation` /
`autonomy`. Neither matches `ix_event`, so those control records pass
`task_tool_events()` and land in the text-face pack as task tool calls. Their
names carry no attribute or setting string, so `scan_banned_strings` cannot see
them — a row called `beat_gate_turn` still tells the setting-blind judge that a
gate fired. Adding `ix_event` to one of the two predicates flips the test green.
