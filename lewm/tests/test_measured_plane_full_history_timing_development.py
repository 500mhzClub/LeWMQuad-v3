"""Complete-population timings, exact state and physical endpoint accounting."""
from copy import deepcopy

import pytest
import cv2

from scripts import measured_plane_full_history_timing_development as comparison
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint
from lewm.tests.test_measured_plane_controller_prefix_runner_development import endpoint


def population(count=8, forecasts=True):
    rows = []
    for frame in range(count):
        calls = int(forecasts and frame >= 3)
        rows.append(dict(frame=frame, execution_order=[0, 1] if frame % 2 == 0 else [1, 0],
            baseline_controller_s=.2, candidate_controller_s=.1,
            complete_original_decision_equal=True, complete_normalized_candidate_equal=True,
            public_inputs_unchanged=True, original_decision_sha256=str(frame),
            baseline_decision_sha256=str(frame), normalized_candidate_decision_sha256=str(frame),
            actual_model_forward_calls=[calls, calls], forecast_compared=bool(calls)))
    states = [dict(frame=f, state_sha256='a'*64, observed_state_equal=True) for f in comparison.state_frames(count)]
    return rows, states


def test_all_observations_and_forecasts_are_separate_complete_populations():
    rows, states = population()
    report = comparison.summarize(rows, states, frames=8, model_sha='model', input_result_sha='result')
    assert report['actual_model_forward_calls'] == [5, 5]
    assert report['timing']['all_observations']['observations'] == 8
    assert report['timing']['forecasts']['observations'] == 5
    assert report['timing']['all_observations']['baseline_over_100ms'] == 8
    assert report['timing']['all_observations']['candidate_over_100ms'] == 0
    assert report['timing']['all_observations']['baseline_total_s'] == pytest.approx(1.6)
    assert report['timing']['all_observations']['candidate_p95_s'] == .1
    assert report['observed_state_checks'] == states
    assert len(report['normalized_state_type_paths']) == 11
    assert report['normalized_state_type_paths'][-1] == 'observer_registration_mission.registration.type'
    assert not report['motion_and_mission_types_normalized']
    assert not report['isolated_benchmark'] and not report['real_time_qualified']
    assert not report['native_execution'] and not report['navigation_outcomes_inferred']


def test_empty_forecast_population_is_explicit_and_final_state_is_always_checked():
    rows, states = population(4, forecasts=False)
    report = comparison.summarize(rows, states, frames=4, model_sha='model', input_result_sha='result')
    assert report['timing']['forecasts']['observations'] == 0
    assert report['timing']['forecasts']['candidate_median_s'] is None
    assert report['timing']['forecasts']['baseline_p95_s'] is None
    assert comparison.state_frames(4) == [0, 3]
    assert comparison.state_frames(4014) == [0, 3, 61, 122, 255, 511, 1023, 2047, 3071, 4013]


@pytest.mark.parametrize('fault', ['short', 'order', 'frame', 'equality', 'input', 'decision',
    'calls', 'bool_calls', 'forecast', 'nan', 'zero', 'bool_time', 'missing_state', 'state_unequal', 'short_sha'])
def test_missing_population_changed_model_calls_or_false_timing_claims_are_rejected(fault):
    rows, states = population()
    if fault == 'short': rows.pop()
    elif fault == 'order': rows[3]['execution_order'] = [0, 1]
    elif fault == 'frame': rows[3]['frame'] = 5
    elif fault == 'equality': rows[3]['complete_normalized_candidate_equal'] = False
    elif fault == 'input': rows[3]['public_inputs_unchanged'] = False
    elif fault == 'decision': rows[3]['normalized_candidate_decision_sha256'] = 'changed'
    elif fault == 'calls': rows[3]['actual_model_forward_calls'] = [1, 0]
    elif fault == 'bool_calls': rows[3]['actual_model_forward_calls'] = [True, True]
    elif fault == 'forecast': rows[3]['forecast_compared'] = False
    elif fault == 'nan': rows[3]['candidate_controller_s'] = float('nan')
    elif fault == 'zero': rows[3]['candidate_controller_s'] = 0
    elif fault == 'bool_time': rows[3]['candidate_controller_s'] = True
    elif fault == 'missing_state': states.pop()
    elif fault == 'state_unequal': states[-1]['observed_state_equal'] = False
    elif fault == 'short_sha': states[-1]['state_sha256'] = 'short'
    with pytest.raises(ValueError): comparison.summarize(rows, states, frames=8, model_sha='model', input_result_sha='result')


def test_original_completed_and_final_partial_commands_are_distinguished():
    tape = []; rows = []
    for frame in range(4):
        row, command = endpoint(frame, dict(requested_command=[0., 0., 0.], terminal=None))
        rows.append(row); tape.append(command)
    for frame, row in enumerate(rows): comparison.reference_endpoint(row, tape, frame, 4)
    tape[-1].update(completed=False, post_sample_index=905)
    comparison.reference_endpoint(rows[-1], tape, 3, 4)
    tape[2].update(completed=False, post_sample_index=855)
    with pytest.raises(ValueError, match='incomplete physical'): comparison.reference_endpoint(rows[2], tape, 2, 4)
    tape = tape[:3]; rows[-1]['decision']['terminal'] = 'COMPLETE'
    comparison.reference_endpoint(rows[-1], tape, 3, 4)
    rows[-1]['decision']['terminal'] = None
    with pytest.raises(ValueError, match='final terminal'): comparison.reference_endpoint(rows[-1], tape, 3, 4)


def test_actual_image_observer_registration_and_mission_state_is_exact_and_sensitive():
    from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
    from lewm.measured_plane_residual_controller_development import MeasuredPlaneResidualController
    from lewm.measured_plane_single_pass_controller_development import MeasuredPlaneSinglePassController
    from lewm.tests.test_measured_plane_comparator_controllers_development import FixedHeadModel, MISSION
    from lewm.tests.test_measured_plane_dual_camera_pose_development import sequence
    from lewm.tests.test_measured_plane_residual_controller_development import move_test_origin
    from scripts.analyze_go2_ground_plane_development_v1 import URDF
    options = dict(public_mission=deepcopy(MISSION), navigation_ticks=40, condition='direct', variant='no_rgb', persistent=True)
    arms = [cls(FixedHeadModel('direct'), ArticulatedCollisionGeometry(URDF), **options)
        for cls in (MeasuredPlaneResidualController, MeasuredPlaneSinglePassController)]
    for original in sequence():
        decisions = []
        for controller in arms:
            p, d, f, kwargs = [move_test_origin(deepcopy(v)) for v in original]
            decisions.append(controller.observe(p, d, f, **kwargs))
        assert all(d['terminal'] is None for d in decisions)
        assert comparison.normalize(decisions[1]) == decisions[0]
        assert fingerprint(comparison.observed_state(arms[0])) == fingerprint(comparison.observed_state(arms[1]))
    failed = [controller.observe(p, d, f, **kwargs) for controller in arms]
    assert all(row['terminal'] == 'SENSOR_OR_MODEL_FAILURE' for row in failed)
    assert comparison.normalize(failed[1]) == failed[0]
    assert fingerprint(comparison.observed_state(arms[0])) == fingerprint(comparison.observed_state(arms[1]))
    before = fingerprint(comparison.observed_state(arms[0]))
    saved = arms[1].motion.model.last_p.copy()
    arms[1].motion.model.last_p[0] += .001
    assert fingerprint(comparison.observed_state(arms[1])) != before
    arms[1].motion.model.last_p[:] = saved
    assert fingerprint(comparison.observed_state(arms[1])) == before
    arms[1].mission.phase = 'SYNTHETIC_CHANGED_PHASE'
    assert fingerprint(comparison.observed_state(arms[1])) != before
    arms[1].mission.phase = arms[0].mission.phase
    assert fingerprint(comparison.observed_state(arms[1])) == before
    arms[1].registration.synthetic_changed_field = True
    assert fingerprint(comparison.observed_state(arms[1])) != before
    del arms[1].registration.synthetic_changed_field
    class UnreviewedRegistration(type(arms[1].registration)): pass
    arms[1].registration = UnreviewedRegistration(identity=(0, 0, 0))
    with pytest.raises(ValueError, match='exact original or existing tiled'):
        comparison.observed_state(arms[1])


@pytest.mark.parametrize('field,value', [('pt', (2., 3.)), ('size', 5.), ('angle', 7.),
    ('response', .5), ('octave', 2), ('class_id', 4)])
def test_every_opencv_keypoint_field_is_preserved_in_recursive_observer_state(field, value):
    point = cv2.KeyPoint(0., 1., 3., 4., .2, 0, 1)
    before = comparison.observer_state_tree({'nested': [point]})
    setattr(point, field, value)
    after = comparison.observer_state_tree({'nested': [point]})
    assert before != after
    assert set(after['nested'][0]['fields']) == {'pt', 'size', 'angle', 'response', 'octave', 'class_id'}
    assert after['nested'][0]['type'] == 'cv2.KeyPoint'
