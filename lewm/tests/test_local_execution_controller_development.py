import math

import numpy as np
import pytest

from lewm.local_execution_controller_development import (
    ARMS, LocalController, continuation_geometry, evaluate_edge, motion_window_ok, trial_spec,
)
from lewm.physical_execution_development import KINDS, WIDTHS
from lewm.tests.test_physical_execution_development import CROSSING, success_arrays


def decide(controller, tick, **overrides):
    values = dict(tick=tick, alignment_error=0., pursuit_error=0., arrival_error=0.,
                  body_forward_velocity=0., angular_velocity=0., crossed=False, stable_arrival=False)
    return controller.decide(**(values | overrides))


def test_fixed_32_trial_population_pairs_only_initial_conditions():
    specs = [trial_spec(kind, width, arm) for kind in KINDS for width in WIDTHS for arm in ARMS]
    assert len({s['scene_id'] for s in specs}) == 32
    for offset in range(0, 32, 4):
        group = specs[offset:offset+4]
        assert len({s['procedural_seed'] for s in group}) == 1
        assert all(s['geometry'] == group[0]['geometry'] for s in group)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('width', WIDTHS)
def test_continuation_moves_boundary_not_scene_or_original(kind, width):
    geometry = trial_spec(kind, width, 'baseline')['geometry']
    second = continuation_geometry(geometry)
    edge = geometry['selected_directed_edge']
    normal = np.array(edge['opening_normal_world'])
    opening = np.array(edge['opening_segment_world'])
    assert second['wall_boxes'] == geometry['wall_boxes']
    assert second['selected_directed_edge']['opening_segment_world'] == pytest.approx(opening + .6 * normal)
    assert second['teacher_route_polyline_world'][-1] == pytest.approx(opening.mean(axis=0) + 1.1 * normal)
    assert geometry == trial_spec(kind, width, 'baseline')['geometry']


@pytest.mark.parametrize('error', [-math.pi, -1.2, -1., 0., .1, 1., 1.2, math.pi])
def test_baseline_matches_frozen_pursuit_formula(error):
    # Literal scalar formula in the frozen execute_teacher implementation.
    expected_vx = 0. if abs(error) >= 1.2 else float(np.clip(.25 * math.cos(error), .08, .25))
    assert decide(LocalController('baseline'), 0, pursuit_error=error) == [expected_vx, 0., float(np.clip(1.5*error, -.45, .45))]


def test_alignment_needs_two_consecutive_stable_observations():
    controller = LocalController('prealign')
    assert decide(controller, 0) == [0., 0., 0.]
    decide(controller, 1, angular_velocity=.3)
    assert decide(controller, 2) == [0., 0., 0.]
    assert decide(controller, 3) == [.25, 0., 0.]
    assert controller.stage == 'APPROACH'


@pytest.mark.parametrize('arm', ARMS)
def test_budget_reserves_five_arrival_ticks(arm):
    controller = LocalController(arm)
    for tick in range(85):
        assert decide(controller, tick, alignment_error=2.) is not None
        if tick >= 80:
            assert controller.stage == 'ARRIVE'
    assert decide(controller, 85) is None
    assert controller.arrival_start_tick == 80


@pytest.mark.parametrize('arm', ARMS)
def test_arrival_never_finishes_before_minimum_five_ticks(arm):
    controller = LocalController(arm)
    for tick in range(2):
        decide(controller, tick)
    for tick in range(2, 7):
        assert decide(controller, tick, crossed=True, stable_arrival=True) is not None
    assert decide(controller, 7, crossed=True, stable_arrival=True) is None


def test_feedback_uses_motion_and_heading_and_clips_request():
    controller = LocalController('arrival_feedback')
    assert decide(controller, 0, crossed=True, body_forward_velocity=1., arrival_error=1.) == [-.08, 0., .45]
    assert decide(controller, 5, crossed=True, stable_arrival=False) is not None
    assert decide(controller, 6, crossed=True, stable_arrival=True) is None


def test_motion_window_requires_all_hundred_samples_not_only_endpoint():
    spec = trial_spec('straight', .75, 'combined')
    arrays = success_arrays()
    poses, twists = arrays['base_pose_world'], arrays['base_twist_world']
    assert not motion_window_ok(poses[:99], twists[:99], spec['geometry'], .75)
    assert motion_window_ok(poses, twists, spec['geometry'], .75)
    twists[-99, 0] = .11
    assert not motion_window_ok(poses, twists, spec['geometry'], .75)


def test_variable_arrival_duration_does_not_relax_motion_thresholds():
    spec = trial_spec('straight', .75, 'combined')
    arrays = success_arrays()
    arrays = {key: np.concatenate([value, value[-50:]]) for key, value in arrays.items()}
    result = evaluate_edge(spec, arrays, stop_reason=None, crossing=CROSSING)
    assert result['status'] == 'SUCCESS'
    assert result['arrival_time_s'] == pytest.approx(.6)
    arrays['base_twist_world'][-1, 0] = .101
    assert evaluate_edge(spec, arrays, stop_reason=None, crossing=CROSSING)['status'] == 'PHYSICAL_FAILURE'


@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_invalid_feedback_is_integrity_failure(value):
    with pytest.raises(ValueError, match='nonfinite'):
        decide(LocalController('baseline'), 0, pursuit_error=value)
