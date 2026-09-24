from lewm.safety import lateral_augmented_state_eligibility_envelope_v1 as subject


def test_fixture():
    assert subject.fixture_payload()["pass"]


def test_contact_before_authority_precedes_persistent():
    assert subject.classify_residual(
        stable_depth=None,
        any_viable_depth=False,
        pre_existing=False,
        contact_before_authority=True,
        predecessor_available=True,
    ) == "CONTACT_BEFORE_CONTROL_AUTHORITY"


def test_primary_classification():
    assert subject.experiment_classification(
        gate_pass=False, recovered=4, persistent=1
    ) == "STATE_ELIGIBILITY_SIGNAL_RESIDUAL_ACTION_SET_NO_GO"
