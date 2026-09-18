"""Reconstruct actual sensor geometry and preserve the original rejection."""
from copy import deepcopy
from functools import partial
import json
import numpy as np
import pytest
from lewm.floor_transport_conflict_readout_development import reconstruct, candidate_residuals, FAILURE
from lewm.floor_pose_registration_development import ROWS, COLUMNS
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from lewm.tests.test_measured_floor_transport_development import item
from lewm.tests.test_frame_floor_cache_development import equal
from scripts import diagnose_go2_direct_flow_maze01_floor_conflict_v1 as runner


@pytest.fixture(autouse=True)
def synthetic_clock(monkeypatch):
    from lewm.tests import test_joint_pulse_execution_development as fixture
    from lewm.tests.test_continuous_pulse_execution_development import visual
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))


@pytest.mark.parametrize('transported_prior', [False, True])
def test_complete_raw_candidate_geometry_reproduces_original_conflict(transported_prior):
    state = MeasuredFloorTransportRegistration(); previous = None
    for frame in range(2 if transported_prior else 1):
        p, d, a, raw, now, image = item(frame, previous, narrow=frame > 0, height=.32)
        prior = state.observe(p, d, a, raw, now_ns=now); previous = raw
    frame += 1
    p, d, a, raw, now, image = item(frame, previous, narrow=True, height=.33)
    inputs = (json.loads(json.dumps(prior)), json.loads(json.dumps(raw)), p, d, a, image)
    before = deepcopy(inputs)
    with pytest.raises(ValueError, match='conflicts with transported'):
        state.observe(p, d, a, raw, now_ns=now)
    report = reconstruct(*inputs, now_ns=now)
    equal(inputs, before)
    assert report['original_registration_rejection_reproduced'] and report['original_failure_latched']
    assert report['correction']['anchor_age_frames'] == frame
    assert report['conflicting_cameras'] == ['auxiliary'] and report['every_current_candidate_retained']
    empty, populated = report['camera_residuals']
    assert empty['count'] == 0 and empty['maximum_residual_m'] is None
    assert populated['over_3mm_count'] == populated['count'] > 0
    assert populated['maximum_residual_m'] > .009
    assert not report['thresholds_changed'] and not report['native_pose_used']


@pytest.mark.parametrize('fault', ['no_conflict', 'wrong_clock', 'bad_anchor', 'wrong_raw_hash'])
def test_reconstruction_cannot_invent_conflict_or_use_stale_witnesses(fault):
    state = MeasuredFloorTransportRegistration()
    p, d, a, raw, now, image = item(height=.32)
    prior = state.observe(p, d, a, raw, now_ns=now)
    p, d, a, raw, now, image = item(1, raw, narrow=True, height=.32 if fault == 'no_conflict' else .33)
    if fault == 'wrong_clock': now += 1
    elif fault == 'bad_anchor': prior['current_pose']['position_initial_body_m'][0] += .01
    elif fault == 'wrong_raw_hash': raw['current_pose']['auxiliary_depth_sha256'] = '0'*64
    with pytest.raises(ValueError):
        reconstruct(json.loads(json.dumps(prior)), json.loads(json.dumps(raw)), p, d, a, image, now_ns=now)


def test_signed_complete_population_and_worst_pixel_are_reported():
    mask = np.zeros((len(ROWS), len(COLUMNS)), bool); mask[0, :2] = True; mask[1, 0] = True
    cloud = np.array([[0., 0., -.32], [1., 0., -.324], [0., 1., -.318]])
    r = candidate_residuals(cloud, mask, np.array([0., 0., 1.]), .32, camera='primary')
    assert r['count'] == 3 and r['over_3mm_count'] == 1
    assert r['worst_candidate']['candidate_index'] == 1
    assert r['worst_candidate']['sampled_row'] == 2 and r['worst_candidate']['sampled_column'] == 6
    assert r['worst_candidate']['signed_residual_m'] < 0
    assert r['maximum_signed_residual_m'] > 0 and r['mean_signed_residual_m'] < 0
    with pytest.raises(ValueError, match='complete ordered'):
        candidate_residuals(cloud[:-1], mask, np.array([0., 0., 1.]), .32, camera='primary')


@pytest.mark.parametrize('fault', [None, 'short', 'earlier_terminal', 'endpoint', 'failure', 'missing_prior', 'nonzero'])
def test_only_complete_prefix_through_first_failure_is_consumed(monkeypatch, fault):
    monkeypatch.setattr(runner, 'BOUNDARY', 2)
    reads = []
    def rows():
        for i in range(3):
            if fault == 'short' and i == 2: return
            if fault == 'earlier_terminal' and i > 1: pytest.fail('read after earlier terminal')
            reads.append(i)
            d = dict(terminal='SENSOR_OR_MODEL_FAILURE' if i == 2 else None,
                failure=FAILURE if i == 2 else None, evidence=None if i == 2 else {'frame':i},
                requested_command=[0., 0., 0.], original_visual_evidence={'frame':i})
            if fault == 'earlier_terminal' and i == 1: d['terminal'] = 'BUDGET'
            if fault == 'failure' and i == 2: d['failure'] = 'different'
            if fault == 'missing_prior' and i == 1: d['evidence'] = None
            if fault == 'nonzero' and i == 2: d['requested_command'] = [.2, 0., 0.]
            yield dict(tick=i, observation_index=i, pre_sample_index=749+50*i+(fault == 'endpoint'), decision=d)
        pytest.fail('read following terminal observation')
    if fault:
        with pytest.raises(ValueError): runner.terminal_pair(rows())
    else:
        prior, terminal = runner.terminal_pair(rows())
        assert prior['tick'] == 1 and terminal['tick'] == 2 and reads == [0, 1, 2]


def admission_fixture():
    prefix = dict(common_prefix_frames=215, first_intervention_frame=214, physical_prefix_samples=11450,
        physical_and_public_prefix_exact=True, all_preintervention_observed_state_exact=True,
        all_preintervention_requested_commands_exact=True, complete_candidate_decisions_match_prospective_prefix=True,
        raw_model_forecast_comparisons=211, all_compared_raw_model_forecasts_exact=True,
        original_intervention_command=[0., 0., 0.], candidate_intervention_command=[0., 0., .45],
        candidate_intervention_command_completed=True, following_physical_outcomes_compared=False,
        unexecuted_outcomes_inferred=False)
    audit = dict(layout_index=1, raw_sensor_reconstruction_pass=True, raw_command_audit_pass=True,
        raw_model_command_replay_pass=True, model_state_unchanged=True, verified_round_trip=False,
        native_evaluation={}, strict_physical_visibility_pass=True, hard_measurement_failed_frames=[], renderer_capture_audit={})
    record = audit|dict(case=runner.CASE[0], status='DIRECT_FLOW_MAZE01_COLLECTED_AND_RAW_AUDITED',
        prefix_comparison=prefix, collection=dict(rgbd_frames=515, completed_ticks=514, schedule_terminal='SENSOR_OR_MODEL_FAILURE'))
    result = dict(status='DIRECT_FLOW_MAZE01_PILOT_V1_COMPLETE', conditions=[record])
    launch = dict(planned_case=list(runner.CASE), implementation_class='DirectFlowFloorTransportController',
        prefix_report={'model_state_sha256':runner.MODEL_STATE})
    return result, launch, audit


@pytest.mark.parametrize('fault', [None, 'partial', 'raw', 'model', 'prefix', 'layout', 'outcome', 'population'])
def test_completed_raw_audit_and_full_physical_prefix_required(fault):
    result, launch, audit = admission_fixture(); record = result['conditions'][0]
    if fault == 'partial': result['status'] = 'COLLECTED'
    elif fault == 'raw': audit['raw_model_command_replay_pass'] = False
    elif fault == 'model': launch['prefix_report']['model_state_sha256'] = '0'*64
    elif fault == 'prefix': record['prefix_comparison']['physical_and_public_prefix_exact'] = False
    elif fault == 'layout': record['layout_index'] = 3
    elif fault == 'outcome': record['verified_round_trip'] = True
    elif fault == 'population': record['collection']['completed_ticks'] = 513
    if fault:
        with pytest.raises(ValueError): runner.admit(result, launch, audit)
    else: runner.admit(result, launch, audit)
