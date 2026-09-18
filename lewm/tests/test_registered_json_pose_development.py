import json
from copy import deepcopy
from functools import partial
import pytest
from lewm.registered_json_pose_development import registered_json_pose


def evidence(monkeypatch):
    from lewm.tests import test_joint_pulse_execution_development as fixture
    from lewm.tests.test_continuous_pulse_execution_development import visual
    from lewm.tests.test_joint_floor_registered_controller_development import packets
    from lewm.joint_floor_registered_evidence_development import JointFloorRegistration
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    p, d, a, raw, now = packets(0, None)
    e = JointFloorRegistration(identity=(0,0,0)).observe(p, d, a, raw, now_ns=now)
    return json.loads(json.dumps(e)), now


def test_json_evidence_validates_without_mutating_record(monkeypatch):
    e, now = evidence(monkeypatch); original = deepcopy(e)
    p, R, pose = registered_json_pose(e, now_ns=now)
    assert pose['frame'] == 0 and p.shape == (3,) and R.shape == (3,3)
    assert e == original and isinstance(e['identity'], list)


@pytest.mark.parametrize('identity',[[True,0,0],[-1,0,0],[0,0],[0,0,1]])
def test_malformed_or_wrong_episode_is_not_normalized_into_validity(monkeypatch,identity):
    e, now = evidence(monkeypatch); e['original_visual_evidence']['identity'] = identity
    with pytest.raises(ValueError): registered_json_pose(e, now_ns=now)


def test_tampered_pose_still_fails_exact_witness_check(monkeypatch):
    e, now = evidence(monkeypatch); e['current_pose']['position_initial_body_m'][0] += .01
    with pytest.raises(ValueError): registered_json_pose(e, now_ns=now)
