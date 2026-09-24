from copy import deepcopy
from functools import partial
import json
from types import SimpleNamespace
import pytest
from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.tests.test_measured_floor_transport_development import item
from lewm.dual_camera_settled_controller_development import DualCameraSettledController
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController
from lewm.measured_floor_transport_prefix_development import normalize_labels, compare_prior, admit_native


def test_complete_public_pipeline_decisions_match_before_transport(monkeypatch):
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    options = dict(public_mission=dict(goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.],
        require_return_after_goal=True), navigation_ticks=100, condition='jepa', variant='full', persistent=True)
    old, new = [cls(None, None, **options) for cls in (DualCameraSettledController, MeasuredFloorTransportController)]
    previous = None
    for frame in range(3):
        p, d, a, raw, now, image = item(frame, previous)
        results = []
        for controller in (old, new):
            controller.motion = SimpleNamespace(observe=lambda *args, **kwargs: raw)
            result = controller.observe(p, d, None, auxiliary_rgb=image, auxiliary_depth=a, now_ns=now)
            assert result['terminal'] is None, result['failure']
            results.append(json.loads(json.dumps(result)))
        saved = dict(tick=frame, decision=results[0]); candidate = results[1]; before = deepcopy(candidate)
        compare_prior(saved, candidate, results[0]['requested_command'], frame=frame)
        assert candidate == before
        for field, value in (('requested_command', [.2, 0., 0.]), ('new_selection', {'hidden_change': True}),
                ('floor_transport_during_missingness_enabled', False), ('controller', 'wrong')):
            bad = deepcopy(candidate); bad[field] = value
            with pytest.raises(ValueError): compare_prior(saved, bad, results[0]['requested_command'], frame=frame)
        previous = raw
    with pytest.raises(ValueError): compare_prior(saved, candidate, [0., 0., 0.], frame=1904)
    candidate['mission_receipt']['observed_settling']['motion_source'] = 'command_extrapolation'
    with pytest.raises(ValueError): normalize_labels(candidate)


def native():
    result = dict(status='DUAL_CAMERA_SETTLED_MAZE_PILOT_V1_COMPLETE', conditions=[dict(case='case',
        status='DUAL_CAMERA_SETTLED_MAZE_COLLECTED_AND_RAW_AUDITED', prefix_comparison=dict(
            physical_and_public_prefix_exact=True, complete_candidate_decisions_match_prospective_prefix=True))])
    report = dict(raw_sensor_reconstruction_pass=True, additional_auxiliary_rgb_reconstructed=True,
        raw_model_command_replay_pass=True, raw_command_audit_pass=True, model_state_unchanged=True)
    return result, report


@pytest.mark.parametrize('fault', ['status', 'case', 'physical_prefix', 'prospective_prefix', 'raw_sensor_reconstruction_pass',
    'additional_auxiliary_rgb_reconstructed', 'raw_model_command_replay_pass', 'raw_command_audit_pass', 'model_state_unchanged'])
def test_incomplete_native_audit_cannot_admit_replay(fault):
    result, report = native(); admit_native(result, report, case='case')
    if fault == 'status': result['status'] = 'COLLECTED'
    elif fault == 'case': result['conditions'][0]['case'] = 'other'
    elif fault == 'physical_prefix': result['conditions'][0]['prefix_comparison']['physical_and_public_prefix_exact'] = False
    elif fault == 'prospective_prefix': result['conditions'][0]['prefix_comparison']['complete_candidate_decisions_match_prospective_prefix'] = False
    else: report[fault] = False
    with pytest.raises(ValueError): admit_native(result, report, case='case')
