from copy import deepcopy
from functools import partial
import json
import numpy as np
import pytest
from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_measured_floor_transport_development import item
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from lewm.measured_floor_transport_readout_development import registered_pose_accuracy


@pytest.fixture
def case(monkeypatch):
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    state = MeasuredFloorTransportRegistration(); rows = []; previous = None
    for frame in range(3):
        p,d,a,raw,now,_ = item(frame,previous,narrow=frame==2,height=.32 if frame==0 else .34)
        e = state.observe(p,d,a,raw,now_ns=now)
        rows.append(dict(tick=frame,decision=dict(evidence=e,original_visual_evidence=raw,terminal=None)))
        previous = raw
    rows = json.loads(json.dumps(rows)); poses = np.zeros((850,7)); poses[:,6] = 1.
    poses[799,:3] = [.01,0.,.02]; poses[849,:3] = [.02,0.,.02]
    return poses, rows


def test_transport_accuracy_uses_actual_endpoint_and_distinct_correction_semantics(case):
    poses,rows = case; before = deepcopy(rows)
    result = registered_pose_accuracy(poses,rows)
    assert result['admitted_pose_frames'] == 3 and result['transported_pose_frames'] == 1
    last = result['records'][-1]
    assert last['pose_admission_kind'] == 'measured_visual_floor_transport' and last['floor_anchor_age_frames'] == 1
    assert last['raw_xyz_error_m'] == pytest.approx(.02) and last['registered_xyz_error_m'] < 1e-6
    assert last['normal_translation_correction_m'] is None and last['normal_alignment_rad'] is None
    assert last['transport_correction_magnitude_m'] == pytest.approx(.02,abs=1e-6)
    assert rows == before
    poses[849,2] = 0.
    changed = registered_pose_accuracy(poses,rows)['records'][-1]
    assert changed['raw_xyz_error_m'] < 1e-12 and changed['registered_xyz_error_m'] == pytest.approx(.02,abs=1e-6)
    assert rows == before and result['native_state_evaluator_only'] and not result['pose_uncertainty_calibrated']


@pytest.mark.parametrize('fault',['anchor_identity','anchor_raw_identity','anchor_pose','raw_copy','pose','endpoint','quaternion'])
def test_tampered_transport_or_missing_actual_endpoint_cannot_report_accuracy(case,fault):
    poses,rows = case; d = rows[2]['decision']; e = d['evidence']; a = e['floor_transport']['anchor']
    if fault == 'anchor_identity': a['identity'][0] = False
    elif fault == 'anchor_raw_identity': a['original_visual_evidence']['identity'][0] = 0.
    elif fault == 'anchor_pose': a['current_pose']['position_initial_body_m'][0] += .01
    elif fault == 'raw_copy': d['original_visual_evidence']['decision_ns'] += 1
    elif fault == 'pose': e['current_pose']['position_initial_body_m'][0] += .01
    elif fault == 'endpoint': poses = poses[:849]
    else: poses[849,6] = .9
    with pytest.raises(ValueError): registered_pose_accuracy(poses,rows)


def test_terminal_missing_pose_is_retained_without_inventing_transport(case):
    poses,rows = case; rows[2]['decision']['evidence'] = None
    with pytest.raises(ValueError): registered_pose_accuracy(poses,rows)
    rows[2]['decision']['terminal'] = 'SENSOR_OR_MODEL_FAILURE'
    r = registered_pose_accuracy(poses,rows)
    assert r['admitted_pose_frames'] == 2 and r['terminal_frames_without_pose'] == [2] and r['transported_pose_frames'] == 0
