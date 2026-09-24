"""Reuse checked map behavior and exercise the complete residual selector chain."""
from copy import deepcopy
from functools import partial
from types import SimpleNamespace
import numpy as np
import pytest

from lewm.residual_current_observation_planning_controller_development import (
    ResidualCurrentObservationPlanningController as New,
    ResidualCurrentObservationPlanningSelector as Selector, METADATA)
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationController as Old, ResidualAnchoredContinuationSelector)
from lewm.current_observation_planning_map_development import CurrentObservationPlanningMap
from lewm.residual_current_observation_planning_prefix_development import compare_step
from lewm.tests.test_current_observation_planning_development import view_fixture, NOW
from lewm.tests.test_measured_floor_transport_development import item
from lewm.tests import test_joint_pulse_execution_development as fixture
from lewm.tests.test_continuous_pulse_execution_development import visual
from lewm.articulated_collision_geometry_development import ArticulatedCollisionGeometry
from lewm.online_executed_residual_development import OnlineExecutedResidual
from scripts.analyze_go2_ground_plane_development_v1 import URDF


def test_existing_map_and_complete_observation_pipeline_are_reused(monkeypatch):
    monkeypatch.setattr(fixture, 'visual', partial(visual, origin=1_500_000_000))
    geometry = ArticulatedCollisionGeometry(URDF)
    options = dict(public_mission=dict(goal_initial_body_xy_m=[1., 0.], return_initial_body_xy_m=[0., 0.],
        require_return_after_goal=True), navigation_ticks=100, condition='jepa', variant='full', persistent=True)
    old, new = Old(None, geometry, **options), New(None, geometry, **options)
    assert type(new.mapper) is CurrentObservationPlanningMap
    assert New.observe is Old.observe and New.advance is Old.advance
    assert new.selector.residual is new.residual and new.memory is new.mapper.surface
    previous = None
    for frame in range(3):
        p, d, a, raw, now, image = item(frame, previous, narrow=frame == 1)
        results = []
        for controller in (old, new):
            controller.motion = SimpleNamespace(observe=lambda *args, **kwargs: raw)
            results.append(controller.observe(p, d, None, now_ns=now, auxiliary_depth=a, auxiliary_rgb=image))
        original, candidate = results
        assert original['terminal'] is None and candidate['terminal'] is None
        check = compare_step(original, candidate, [0., 0., 0.], frame=frame)
        assert not check['requested_command_changed'] and not check['terminal_changed']
        normalized = {k:v for k,v in candidate.items() if k not in METADATA}
        normalized['controller'] = original['controller']
        assert normalized == original
        assert new.mapper.floor == old.mapper.floor and new.mapper.occupied == old.mapper.occupied
        for xy, yaw in (([0., 0.], 0.), ([.01, 0.], .04)):
            assert new.memory.footprint(geometry, xy, yaw, now_ns=now, persistent=True) == old.memory.footprint(
                geometry, xy, yaw, now_ns=now, persistent=True)
        previous = raw
    result = new.observe({}, {}, {}, now_ns=1)
    assert result['terminal'] == 'SENSOR_OR_MODEL_FAILURE' and result['requested_command'] == [0., 0., 0.]
    assert not result['memoryless_controller']


def test_all_residual_feasibility_layers_receive_current_cells_and_retained_contact(monkeypatch):
    from lewm import mission_target_waypoint_selection_development as base
    from lewm.observation_horizon_predictive_selection_development import score_candidates
    owner, view, _ = view_fixture(); contacts = []
    owner.surface.last_ns = NOW
    def footprint(geometry, xy, yaw, *, now_ns, persistent):
        contacts.append(persistent); return dict(possible_intersection=False)
    owner.surface.footprint = footprint
    owner.planning_view = lambda *, now_ns: view
    forecast = np.zeros((6, 8, 5)); forecast[:, :, 3] = 1.; forecast[:, :, 4] = -10.
    forecast[1, :, 0] = np.linspace(.01, .1, 8)
    def select(model, history, *, head, input_variant, goal_body_xy_m, contact_penalty_m):
        return score_candidates(forecast, goal_body_xy_m=goal_body_xy_m, contact_penalty_m=contact_penalty_m) | dict(
            prediction=forecast.tolist(), first_prediction_horizon_ns=100_000_000,
            target_offsets_ns=list(range(100_000_000, 800_000_001, 100_000_000)), selection_wall_ms=0.)
    monkeypatch.setattr(base, 'select', select)
    residual = OnlineExecutedResidual(); residual.now_ns = NOW; residual.frame = 1
    selector = Selector(residual=residual, condition='jepa', variant='full', goal_initial_body_xy_m=[1., 0.])
    assert Selector.__bases__ == (ResidualAnchoredContinuationSelector,)
    result = selector.choose(None, None, owner, object(), now_ns=NOW)
    assert result['planning_map_receipt'] == view.receipt
    assert result['proposal']['observed_floor_cells'] == len(view.floor)
    assert result['proposal']['occupied_cells'] == 0 and owner.occupied
    assert all(s['minimum_observed_cell_distance_m'] is None
        for check in result['nominal_path_checks'] for s in check['segments'])
    assert contacts and all(contacts)
    result['planning_map_receipt']['camera_witnesses'].clear()
    assert len(view.receipt['camera_witnesses']) == 2


def pair():
    original = dict(controller='residual_anchored_continuation_controller_v1', tick=3,
        evidence={}, original_visual_evidence={}, memory_receipt={}, mission_receipt={},
        observed_goal_distance_m=1., auxiliary_floor_partition_receipt={},
        causal_residual_receipt=dict(pending_forecast_tick=3, residuals=[]),
        requested_command=[0., 0., .45], terminal=None,
        new_selection=dict(prediction=[[[0.]]]))
    candidate = deepcopy(original) | dict(controller='residual_current_observation_planning_controller_v1', **METADATA)
    candidate['new_selection']['planning_map_receipt'] = dict(frame=3, measured_ns=1_800_000_000,
        planning_map_variant='current_paired_observation', accumulated_planning_cells_queried=False,
        persistent_contact_history_retained=True)
    return original, candidate


@pytest.mark.parametrize('fault', ['pose', 'mission', 'residual', 'forecast', 'actual', 'scope', 'clock', 'pending'])
def test_prefix_rejects_changes_outside_planning_scope(fault):
    original, candidate = pair(); actual = original['requested_command']
    if fault == 'pose': candidate['evidence']['changed'] = True
    elif fault == 'mission': candidate['mission_receipt']['changed'] = True
    elif fault == 'residual': candidate['causal_residual_receipt']['residuals'].append({})
    elif fault == 'forecast': candidate['new_selection']['prediction'] = [[[1.]]]
    elif fault == 'actual': actual = [0., 0., 0.]
    elif fault == 'scope': candidate['memoryless_controller'] = True
    elif fault == 'clock': candidate['new_selection']['planning_map_receipt']['measured_ns'] += 1
    else: candidate['causal_residual_receipt']['pending_forecast_tick'] = 4
    with pytest.raises(ValueError): compare_step(original, candidate, actual, frame=3)


def test_prefix_records_changed_request_or_terminal_without_inferring_outcomes():
    original, candidate = pair()
    candidate['requested_command'] = [.16, 0., .45]
    result = compare_step(original, candidate, original['requested_command'], frame=3)
    assert result['requested_command_changed'] and result['raw_model_forecasts_exact']
    candidate['requested_command'] = original['requested_command']
    candidate['terminal'] = 'STOP'; candidate['causal_residual_receipt']['pending_forecast_tick'] = None
    result = compare_step(original, candidate, original['requested_command'], frame=3)
    assert result['terminal_changed'] and not result['requested_command_changed']
