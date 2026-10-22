"""The simulator's copy of the retired-setting table must match the authoring
schema's. `simulator/` never imports `scripts/` at runtime, so the two dicts
are hand-maintained; this test is the only thing that notices a divergence."""

from scripts.outline_schema import RETIRED_SETTINGS as AUTHORING_RETIRED
from simulator.schemas import RETIRED_SETTINGS as SIMULATOR_RETIRED


def test_the_simulator_retired_settings_mirror_the_authoring_schema() -> None:
    assert set(SIMULATOR_RETIRED) == set(AUTHORING_RETIRED), (
        "retired (attribute, setting) keys differ between simulator/schemas.py "
        "and scripts/outline_schema.py — every retirement lands in both"
    )
    for key, reason in AUTHORING_RETIRED.items():
        assert SIMULATOR_RETIRED[key] == reason, f"reason text differs for {key}"
