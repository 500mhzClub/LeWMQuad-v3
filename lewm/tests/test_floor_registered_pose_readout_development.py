from copy import deepcopy
from functools import partial
import json
import numpy as np
import pytest
from lewm.floor_registered_evidence_development import FloorRegistration
from lewm.joint_floor_registered_evidence_development import JointFloorRegistration
from lewm.floor_registered_pose_readout_development import registered_pose_accuracy
from lewm.tests.test_floor_registered_controller_development import packets
from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual


@pytest.fixture(params=[FloorRegistration, JointFloorRegistration])
def case(monkeypatch, request):
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    state = request.param(); rows = []; previous = None
    for frame in range(2):
        p, d, a, raw, now = packets(frame, previous)
        e = state.observe(p, d, a, raw, now_ns=now)
        rows.append(dict(tick=frame, decision=dict(evidence=e, original_visual_evidence=raw, terminal=None)))
        previous = raw
    rows = json.loads(json.dumps(rows))
    poses = np.zeros((800, 7)); poses[:, 6] = 1.
    poses[799, :3] = [.01, 0., .02]
    return poses, rows


def test_separate_original_and_registered_accuracy_uses_actual_current_endpoint(case):
    poses, rows = case; before = deepcopy(rows)
    result = registered_pose_accuracy(poses, rows)
    assert rows == before
    assert result['admitted_pose_frames'] == 2
    last = result['records'][-1]
    assert last['raw_xyz_error_m'] == pytest.approx(.02)
    assert last['registered_xyz_error_m'] < 1e-6
    assert result['native_state_evaluator_only'] and not result['pose_uncertainty_calibrated']
    # Changing the actual measured endpoint changes errors, never either pose.
    poses[799, 2] = 0.
    changed = registered_pose_accuracy(poses, rows)['records'][-1]
    assert changed['raw_xyz_error_m'] < 1e-12
    assert changed['registered_xyz_error_m'] == pytest.approx(.02, abs=1e-6)
    assert rows == before


@pytest.mark.parametrize('fault', ['clock', 'identity', 'nested_identity', 'pose', 'raw_copy', 'endpoint', 'quaternion'])
def test_malformed_record_or_missing_physics_cannot_report_accuracy(case, fault):
    poses, rows = case
    if fault == 'clock': rows[1]['tick'] = 2
    elif fault == 'identity': rows[1]['decision']['evidence']['identity'][0] = False
    elif fault == 'nested_identity': rows[1]['decision']['evidence']['original_visual_evidence']['identity'][0] = 0.
    elif fault == 'pose': rows[1]['decision']['evidence']['current_pose']['position_initial_body_m'][2] += .01
    elif fault == 'raw_copy': rows[1]['decision']['original_visual_evidence']['decision_ns'] += 1
    elif fault == 'endpoint': poses = poses[:799]
    else: poses[799, 6] = .9
    with pytest.raises(ValueError): registered_pose_accuracy(poses, rows)


def test_terminal_missing_pose_is_explicit_and_nonterminal_missing_pose_rejected(case):
    poses, rows = case
    rows[1]['decision']['evidence'] = None
    with pytest.raises(ValueError): registered_pose_accuracy(poses, rows)
    rows[1]['decision']['terminal'] = 'SENSOR_OR_MODEL_FAILURE'
    result = registered_pose_accuracy(poses, rows)
    assert result['admitted_pose_frames'] == 1 and result['terminal_frames_without_pose'] == [1]
    with pytest.raises(ValueError): registered_pose_accuracy(poses[:799], rows)
