"""Numerical comparison with actual production scorers on synthetic motion."""
from contextlib import contextmanager
from copy import deepcopy
import io
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lewm import independent_tracking_numerical_verification_development as check
from lewm.independent_tracking_challenge_development import SEGMENTS, MAX_TICKS
from lewm.independent_tracking_coverage_development import measured_coverage
from lewm.tests.test_independent_tracking_challenge_development import native_fixture
from lewm.tests.test_independent_tracking_stress_cohort_development import prepared, template, raw_fixture


def coverage_pair(direction='left', *, ticks=442, extra=0, samples=None, stop=None, acquisition=None):
    t, p, v = native_fixture(direction)
    n = 750 + ticks * 50 + extra if samples is None else samples
    raw = dict(timestamp_s=t[:n], base_pose_world=p[:n], base_twist_world=v[:n])
    result = dict(completed_ticks=ticks, schedule_complete=ticks == 442 and stop is None and acquisition is None,
        physical_stop=stop, acquisition_stop=acquisition)
    saved = measured_coverage(direction, *raw.values(), **result)
    return raw, result, saved


def test_independent_schedule_transcription_matches_current_fixed_experiment():
    assert tuple(s[0] for s in SEGMENTS) == check.SEGMENTS
    assert (0, *np.cumsum([s[1] for s in SEGMENTS])) == check.BOUNDARIES
    assert MAX_TICKS == 442


@pytest.mark.parametrize('direction', ['left', 'right'])
def test_full_measured_motion_reconstruction(direction):
    raw, result, saved = coverage_pair(direction)
    r = check.verify_coverage(raw, result, saved, direction=direction)
    assert r['coverage']['intended_motion_covered'] and r['numerical_coverage_reconstruction_verified']
    assert not r['raw_artifacts_authenticated_by_interface'] and not r['full_challenge_pass']


@pytest.mark.parametrize('samples', [0, 1, 749, 750])
def test_early_stop_is_retained_without_fictitious_future_coverage(samples):
    raw, result, saved = coverage_pair(ticks=0, samples=samples, stop='SETTLING_STOP')
    r = check.verify_coverage(raw, result, saved, direction='left')
    assert not r['coverage']['intended_motion_covered']
    assert r['coverage']['final_second_linear_speed_max_m_s'] is None


@pytest.mark.parametrize('extra', [0, 1, 49, 50])
def test_interrupted_command_population_and_exact_acquisition_boundary(extra):
    raw, result, saved = coverage_pair(ticks=100, extra=extra, stop='CONTACT_STOP')
    assert not check.verify_coverage(raw, result, saved, direction='left')['coverage']['intended_motion_covered']
    result['physical_stop'] = None; result['acquisition_stop'] = 'RGBD_MISSING'
    if extra:
        with pytest.raises(ValueError, match='sample population'):
            check.reconstruct_coverage(raw, direction='left', **result)
    else:
        expected = check.reconstruct_coverage(raw, direction='left', **result)
        assert expected['acquisition_stop'] == 'RGBD_MISSING'


@pytest.mark.parametrize('fault', ['stationary', 'wrong_turn', 'one_stop_sample', 'path_not_displacement'])
def test_completed_commands_alone_do_not_prove_physical_execution(fault):
    raw, result, _ = coverage_pair()
    if fault == 'stationary': raw['base_pose_world'][:, :3] = 0.; raw['base_pose_world'][:, 3:] = [0., 0., 0., 1.]
    elif fault == 'wrong_turn': raw['base_pose_world'][:, 5] *= -1
    elif fault == 'one_stop_sample': raw['base_twist_world'][-499, 0] = .021
    else:
        # Long travel that returns to its departure cannot be called translation.
        start, end = 749 + 20 * 50, 749 + 60 * 50
        raw['base_pose_world'][start:end + 1, 0] = np.sin(np.linspace(0, 2 * np.pi, end - start + 1))
    saved = measured_coverage('left', *raw.values(), **result)
    r = check.verify_coverage(raw, result, saved, direction='left')
    assert r['coverage']['schedule_complete'] and not r['coverage']['intended_motion_covered']
    if fault == 'path_not_displacement':
        assert r['coverage']['segments']['approach']['planar_path_m'] > 3.
        assert not r['coverage']['translation_coverage']['approach']


@pytest.mark.parametrize('fault', ['flag', 'metric', 'direction', 'clock', 'invalid_quaternion', 'nan_twist', 'truncation'])
def test_coverage_verification_rejects_corruption(fault):
    raw, result, saved = coverage_pair()
    if fault == 'flag': saved['intended_motion_covered'] = False
    elif fault == 'metric': saved['segments']['turn_out']['signed_yaw_rad'] += .001
    elif fault == 'direction': saved['direction'] = 'right'
    elif fault == 'clock': raw['timestamp_s'][3] += .001
    elif fault == 'invalid_quaternion': raw['base_pose_world'][3, 6] = .5
    elif fault == 'nan_twist': raw['base_twist_world'][3, 0] = np.nan
    elif fault == 'truncation': raw = {k: v[:-1] for k, v in raw.items()}
    with pytest.raises(ValueError): check.verify_coverage(raw, result, saved, direction='left')


@pytest.mark.parametrize('angle', [0., 1e-12, .5, math.pi / 2, math.pi - 1e-10, math.pi])
def test_matrix_angle_handles_small_angles_and_near_pi(angle):
    axis = np.array([1., 2., 3.]); axis /= np.linalg.norm(axis)
    r = Rotation.from_rotvec(angle * axis)
    assert check._angle(r.as_matrix()) == pytest.approx(r.magnitude(), abs=1e-12)


@pytest.fixture
def pose_case(tmp_path, monkeypatch):
    import scripts.independent_tracking_evaluation_development as scorer
    frames = 12; n = 750 + (frames - 1) * 50
    t = np.arange(1, n + 1) * .002
    motion = np.maximum(np.arange(n) - 749, 0.) / 500
    origin = Rotation.from_euler('xyz', [.2, -.3, .7])
    orientations = origin * Rotation.from_rotvec(motion[:, None] * np.array([.3, .2, 1.7]))
    xyz = np.column_stack([motion, motion ** 2, .02 * motion]) + [1.7, -2.3, .4]
    native = np.column_stack([xyz, orientations.as_quat()])
    raw = dict(timestamp_s=t, base_pose_world=native)
    rows = []; evaluation = []; categories = dict(both=0, original_only=0, temporal_anchor_only=0, neither=0)
    for frame in range(frames):
        index = 749 + 50 * frame; now = 1_500_000_000 + frame * 100_000_000
        p = origin.inv().apply(native[index, :3] - native[749, :3])
        r = origin.inv() * orientations[index]; arms = {}
        for arm in check.ARMS:
            pose = None if arm == 'original' and frame >= 8 else dict(frame=frame, measured_ns=now,
                global_history_reset=False, position_error_bound=None,
                position_initial_body_m=(p + [.001 * frame, .002, -.001]).tolist(),
                rotation_initial_body_from_current_body=(r * Rotation.from_rotvec([.001 * frame, -.002, .001])).as_matrix().tolist())
            arms[arm] = dict(pose=pose, observer_wall_ms=float(frame + 1),
                continuity=dict(status='MEASURED_INCREMENT_BRIDGE' if frame in (4, 5) else 'ANCHOR_MEASUREMENT'))
        category = 'both' if frame < 8 else 'temporal_anchor_only'; categories[category] += 1
        rows.append(dict(frame=frame, measured_ns=now, arms=arms, availability=category))
    report = dict(frames=frames, estimates_file='synthetic.jsonl', availability=categories,
        arms=dict(original=dict(frames=frames, available=8, first_failure=8),
            temporal_anchor=dict(frames=frames, available=12, first_failure=None)), continuity=dict(total_bridged_frames=2))
    path = tmp_path / 'synthetic.jsonl'; path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    monkeypatch.setattr(scorer, 'artifact_path', lambda output, name: path)
    class Store:
        output = tmp_path
        @contextmanager
        def stream(self, name): yield io.StringIO()
        def append(self, target, row): evaluation.append(deepcopy(row))
    score = scorer._score_trial(Store(), 'synthetic', report, raw)
    return raw, rows, evaluation, report, score


def test_independent_pose_math_matches_real_scorer_on_noncommuting_3d_motion(pose_case):
    raw, rows, evaluation, report, score = pose_case
    result = check.verify_pose_stream(raw, iter(rows), iter(evaluation), report, score)
    assert result['frames'] == 12 and result['availability']['temporal_anchor_only'] == 4
    assert result['score']['arms']['original']['paired_position_m']['count'] == 8
    assert result['score']['bridged_frame_errors']['position_m']['count'] == 2
    assert result['score']['empirical_local_pose_allocation_met']['temporal_anchor']
    assert not result['score']['empirical_local_pose_allocation_met']['original']
    assert not result['full_challenge_pass'] and not result['stress_application_verified']


@pytest.mark.parametrize('fault', ['evaluation_length', 'estimate_length', 'clock', 'row_error', 'saved_mean',
    'saved_count', 'paired_count', 'bridge_count', 'resurrection', 'reflection', 'nan_position',
    'negative_time', 'availability', 'invented_bound', 'reset', 'promoted', 'unbounded_frame_count'])
def test_pose_verification_rejects_false_scores_and_missing_denominators(pose_case, fault):
    raw, rows, evaluation, report, score = deepcopy(pose_case)
    if fault == 'evaluation_length': evaluation.pop()
    elif fault == 'estimate_length': rows.pop()
    elif fault == 'clock': rows[1]['measured_ns'] += 1
    elif fault == 'row_error': evaluation[3]['errors']['temporal_anchor']['position_m'] += .001
    elif fault == 'saved_mean': score['arms']['temporal_anchor']['position_m']['mean'] += .001
    elif fault == 'saved_count': score['arms']['temporal_anchor']['position_m']['count'] -= 1
    elif fault == 'paired_count': score['arms']['original']['paired_position_m']['count'] += 1
    elif fault == 'bridge_count': report['continuity']['total_bridged_frames'] = 1
    elif fault == 'resurrection': rows[9]['arms']['original']['pose'] = deepcopy(rows[9]['arms']['temporal_anchor']['pose']); rows[9]['availability'] = 'both'
    elif fault == 'reflection': rows[2]['arms']['temporal_anchor']['pose']['rotation_initial_body_from_current_body'] = np.diag([1., 1., -1.]).tolist()
    elif fault == 'nan_position': rows[2]['arms']['temporal_anchor']['pose']['position_initial_body_m'][0] = np.nan
    elif fault == 'negative_time': rows[2]['arms']['original']['observer_wall_ms'] = -1.
    elif fault == 'availability': rows[2]['availability'] = 'neither'
    elif fault == 'invented_bound': rows[2]['arms']['temporal_anchor']['pose']['position_error_bound'] = 0.
    elif fault == 'reset': rows[2]['arms']['temporal_anchor']['pose']['global_history_reset'] = True
    elif fault == 'promoted': score['navigation_qualified'] = True
    elif fault == 'unbounded_frame_count': report['frames'] = 444
    with pytest.raises(ValueError): check.verify_pose_stream(raw, rows, evaluation, report, score)


def test_scalar_tolerance_never_allows_a_pass_flag_change():
    with pytest.raises(ValueError): check.same({'pass': True}, {'pass': False}, tolerance=100.)
    with pytest.raises(ValueError): check.same(1, True)
    with pytest.raises(ValueError): check.same(True, 1.)


def test_statistics_use_complete_population_and_linear_percentiles():
    values = [4., 0., 2., 7., 1.]
    actual = check._statistics(values)
    assert actual['mean'] == np.mean(values) and actual['p95'] == np.quantile(values, .95)
    assert check._statistics([]) == dict(count=0, mean=None, median=None, p95=None, maximum=None)


def test_global_heading_wrap_and_quaternion_sign_do_not_change_coverage():
    raw, result, _ = coverage_pair('right')
    pose = raw['base_pose_world']; rotation = Rotation.from_euler('z', math.radians(170))
    pose[:, :3] = rotation.apply(pose[:, :3]) + [12., -7., .4]
    pose[:, 3:] = (rotation * Rotation.from_quat(pose[:, 3:])).as_quat()
    pose[::2, 3:] *= -1
    saved = measured_coverage('right', *raw.values(), **result)
    assert check.verify_coverage(raw, result, saved, direction='right')['coverage']['intended_motion_covered']


def test_empty_pose_population_retained_but_never_meets_allocation():
    raw = dict(timestamp_s=np.empty(0), base_pose_world=np.empty((0, 7)))
    availability = dict(both=0, original_only=0, temporal_anchor_only=0, neither=0)
    report = dict(frames=0, availability=availability, continuity=dict(total_bridged_frames=0),
        arms={a: dict(frames=0, available=0, first_failure=None) for a in check.ARMS})
    empty = check._statistics([])
    score = dict(frames=0, availability=availability,
        arms={a: {k: dict(empty) for k in (*check.METRICS, 'paired_position_m', 'paired_orientation_rad',
            'observer_wall_ms')} for a in check.ARMS}, bridged_frame_errors={k: dict(empty) for k in check.METRICS},
        empirical_local_pose_allocation_met=dict.fromkeys(check.ARMS, False),
        empirical_allocation_is_calibrated_bound=False, incremental_error_is_consecutive_estimate_difference=True,
        observer_timing_excludes_acquisition_and_control=True, navigation_qualified=False)
    verified = check.verify_pose_stream(raw, [], [], report, score)
    assert verified['frames'] == 0 and not any(verified['score']['empirical_local_pose_allocation_met'].values())


def test_all_eight_base_and_88_stress_scores_from_real_observers_match_independent_math(prepared, monkeypatch):
    """Three-frame synthetic tapes: actual schema/observers, no fault onset or native simulator."""
    import scripts.independent_tracking_stress_cohort_development as stress
    import scripts.independent_tracking_cohort_development as cohort
    import scripts.independent_tracking_evaluation_development as scorer
    from lewm.independent_tracking_challenge_development import TRIALS
    store, c, b, s, phase = prepared
    visited = []; monkeypatch.setattr(scorer, '_raw_audit', raw_fixture(visited))
    result = stress.evaluate_complete_population(store, c, b, s, 'a' * 64)
    base = cohort.read(store.output, 'sensor_phase_complete.json')
    verified = []
    for trial in TRIALS:
        n = store.episodes[trial]['result']['physics_samples']
        poses = np.zeros((n, 7)); poses[:, 6] = 1.
        raw = dict(timestamp_s=np.arange(1, n + 1) * .002, base_pose_world=poses)
        for scenario in (None, *stress.SCENARIOS):
            report = base['reports'][trial] if scenario is None else phase['reports'][trial][scenario]
            score = result['scores'][trial] if scenario is None else result['stress_scores'][trial][scenario]
            name = trial + '_evaluation.jsonl' if scenario is None else stress.stream_name(trial, scenario, evaluation=True)
            row = check.verify_pose_stream(raw, stress.rows(store.output, report['estimates_file']),
                stress.rows(store.output, name), report, score)
            verified.append((trial, scenario, row['frames']))
            assert not row['full_challenge_pass'] and not row['stress_application_verified']
    assert len(verified) == 96 and sum(v[2] for v in verified) == 288
    assert visited == list(TRIALS)
