from lewm.control import lateral_recovery_locomotion_controller_dev_v1 as subject


def test_frozen_contract():
    assert subject.fixture()["pass"]
    assert subject.CONTINUATION_UPDATES == 125


def test_contract_is_deterministic():
    assert subject.contract() == subject.contract()
