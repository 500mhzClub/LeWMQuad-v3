import json
from copy import deepcopy
import numpy as np
import pytest
from lewm.recorded_joint_pose_admission_development import recorded_joint_pose
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.causal_sensor_state import SensorContractError
from lewm.tests.test_joint_pulse_execution_development import joint_visual


def test_json_round_trip_restores_only_identity_encoding_and_preserves_pose_witness():
    first, _ = joint_visual(0)
    for e, now in (joint_visual(0), joint_visual(1, (.01, -.02, .03), previous=first)):
        raw = json.loads(json.dumps(e)); before = deepcopy(raw)
        with pytest.raises(SensorContractError): current_joint_pose(raw, identity=(0, 0, 0), now_ns=now)
        a = recorded_joint_pose(raw, identity=(0, 0, 0), now_ns=now)
        b = current_joint_pose(e, identity=(0, 0, 0), now_ns=now)
        np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
        assert a[2] == b[2] and raw == before


@pytest.mark.parametrize('fault', ['episode', 'bool', 'float', 'clock', 'witness'])
def test_original_identity_time_and_pose_witness_gates_remain_active(fault):
    first, _ = joint_visual(0); e, now = joint_visual(1, previous=first)
    r = json.loads(json.dumps(e))
    if fault == 'episode': r['identity'][1] = 1
    elif fault == 'bool': r['identity'][0] = False
    elif fault == 'float': r['identity'][0] = 0.
    elif fault == 'clock': r['decision_ns'] += 1
    else: r['current_pose']['position_initial_body_m'][0] = .1
    with pytest.raises(SensorContractError): recorded_joint_pose(r, identity=(0, 0, 0), now_ns=now)
