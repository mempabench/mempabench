"""Throwaway `memory.turn_journal` stand-ins, for the step-9 stub proof only.

`tests/test_harness/test_resume_pool.py` is the acceptance suite for plan §9
step 9 and cannot run until `memory/turn_journal.py` exists, so on this branch
it is skipped at import. These two modules make the claim "those tests assert
behaviour, not the module's presence" re-verifiable instead of asserted in a
receipt: `test_resume_pool_stub_proof.py` loads the acceptance suite twice, once
against each stub, and shows the suite passing against the faithful one and
failing on its own assertions against the inert one.

Neither module is production code and neither is what step 9 should land: the
faithful one is the smallest journal that satisfies the suite, written to prove
the suite is satisfiable at all.
"""
